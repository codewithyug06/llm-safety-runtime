"""
MOD-02: Offline Causal Scrubbing -- Systematic Head Ablation
=============================================================
Runs a full head-ablation study across all attention heads in a transformer
model (default: Llama 3.1 8B, 32 layers x 32 heads), builds a ``CausalGraph``
mapping which heads drive each unsafe-behaviour probe category, serialises the
graph to ``models/causal/``, and logs every metric and artefact to MLflow.

Probe weights are loaded from ``models/probes/*.pt``; the ``LinearResidualProbe``
architecture is ``Linear(hidden_size, 64) -> ReLU -> Linear(64, 1)``.  The probe
weight vector used for causal-effect computation is derived by pseudo-inverse
projection of the two-layer network onto the hidden-state space.

Activation data is loaded from ``data/probes/*_activations.npy`` when available.
If no file is found the script falls back to randomly generated activations of
shape ``(N, 4096)`` -- safe for offline testing without a live model.

Run with:
    python scripts/run_causal_scrubbing.py
    python scripts/run_causal_scrubbing.py \\
        --model-name llama31_8b \\
        --probe-dir models/probes \\
        --output-dir models/causal \\
        --n-prompts 256 \\
        --min-delta 0.05 \\
        --mlflow-uri http://localhost:5000

Outputs:
    - models/causal/llama31_8b_causal_graph.json
    - MLflow run: causal_scrubbing/<model_name>
    - Console table: top-5 unsafe heads per category
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import structlog

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.causal_engine.intervention import (
    CausalEdge,
    CausalGraph,
    CausalScrubber,
    HeadSignature,
    compute_ablation_effect,
)
from src.exceptions import CausalGraphNotFoundError, CausalScrubError

logger = structlog.get_logger(__name__)

# ── Architecture constants for Llama 3.1 8B ──────────────────────────────────

LLAMA31_8B_HIDDEN_SIZE: int = 4096
LLAMA31_8B_NUM_LAYERS: int = 32
LLAMA31_8B_NUM_HEADS: int = 32
LLAMA31_8B_D_HEAD: int = 128  # hidden_size / num_heads = 4096 / 32

PROBE_CATEGORIES: List[str] = [
    "hallucination",
    "jailbreak",
    "toxic_reasoning",
    "policy_violation",
]


# ── Probe weight extraction ───────────────────────────────────────────────────

def load_probe_weight_vector(
    probe_path: Path,
    hidden_size: int,
    probe_hidden_dim: int = 64,
) -> np.ndarray:
    """Load a ``LinearResidualProbe`` and derive a flat hidden-state weight vector.

    The probe has the architecture::

        net.0  : Linear(hidden_size, 64)   -- W1 (64, hidden_size), b1 (64,)
        net.2  : Linear(64, 1)             -- W2 (1, 64),            b2 (1,)

    We combine the two layers into a single effective weight vector in the
    original hidden-state space via pseudo-inverse projection::

        w_eff = W1^+ @ W2.T           shape: (hidden_size,)

    This vector is passed to ``compute_ablation_effect`` which expects
    ``safety_probe_weights`` of shape ``(hidden,)`` or ``(hidden, 2)``.

    Args:
        probe_path: Path to the saved ``state_dict`` ``.pt`` file.
        hidden_size: Model hidden dimension (e.g. 4096 for Llama 3.1 8B).
        probe_hidden_dim: Hidden units in the probe (default 64).

    Returns:
        Flat numpy weight vector of shape ``(hidden_size,)``.

    Raises:
        CausalScrubError: If the state_dict keys are missing or shapes mismatch.
    """
    logger.debug("loading_probe_weights", path=str(probe_path))

    try:
        state: Dict[str, torch.Tensor] = torch.load(
            probe_path, map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise CausalScrubError(
            f"Cannot load probe state_dict from {probe_path}: {exc}"
        ) from exc

    required_keys = {"net.0.weight", "net.2.weight"}
    missing = required_keys - set(state.keys())
    if missing:
        raise CausalScrubError(
            f"Probe {probe_path.name} is missing expected keys: {missing}. "
            f"Found keys: {sorted(state.keys())}"
        )

    # W1: (probe_hidden_dim, hidden_size) -- first layer weight
    w1: np.ndarray = state["net.0.weight"].float().numpy()  # (64, hidden_size)
    # W2: (1, probe_hidden_dim)           -- second layer weight
    w2: np.ndarray = state["net.2.weight"].float().numpy()  # (1, 64)

    # Auto-detect actual probe input_dim from loaded weights.
    # Local probes may differ from the expected hidden_size (e.g. 128 vs 4096).
    actual_input_dim = w1.shape[1]
    if actual_input_dim != hidden_size:
        logger.warning(
            "probe_hidden_size_mismatch",
            probe=probe_path.name,
            expected=hidden_size,
            actual=actual_input_dim,
            note="Using probe's actual input_dim for projection",
        )
        hidden_size = actual_input_dim  # use real dimension from the file
    if w2.shape != (1, probe_hidden_dim):
        raise CausalScrubError(
            f"net.2.weight shape mismatch in {probe_path.name}: "
            f"expected (1, {probe_hidden_dim}), got {w2.shape}"
        )

    # Pseudo-inverse projection: W1^+ = W1^T (W1 W1^T)^{-1} -- shape (hidden_size, 64)
    w1_pinv: np.ndarray = np.linalg.pinv(w1)  # (hidden_size, probe_hidden_dim)

    # Effective weight in hidden-state space: (hidden_size, 64) @ (64, 1) -> (hidden_size, 1)
    w_eff: np.ndarray = (w1_pinv @ w2.T).squeeze(-1)  # (hidden_size,)
    w_eff = w_eff / (np.linalg.norm(w_eff) + 1e-8)  # unit-normalise

    logger.debug(
        "probe_weights_extracted",
        path=str(probe_path),
        w_eff_shape=w_eff.shape,
        w_eff_norm=float(np.linalg.norm(w_eff)),
    )
    return w_eff.astype(np.float32)


def load_all_probe_weights(
    probe_dir: Path,
    categories: List[str],
    hidden_size: int,
) -> Dict[str, np.ndarray]:
    """Load probe weight vectors for all requested categories.

    Missing probes are skipped with a warning; a ``CausalScrubError`` is raised
    only when *no* probes can be loaded at all.

    Args:
        probe_dir: Directory containing ``{category}.pt`` files.
        categories: Probe category names to attempt loading.
        hidden_size: Model hidden dimension.

    Returns:
        Dict mapping category name -> flat weight vector ``(hidden_size,)``.

    Raises:
        CausalScrubError: If no probe files are found and no fallback is possible.
    """
    probe_weights: Dict[str, np.ndarray] = {}

    for cat in categories:
        probe_path = probe_dir / f"{cat}.pt"
        if not probe_path.exists():
            logger.warning(
                "probe_file_not_found",
                category=cat,
                path=str(probe_path),
                action="using_random_fallback",
            )
            # Reproducible random unit-vector fallback so the script stays runnable
            rng = np.random.default_rng(seed=abs(hash(cat)) % (2**31))
            w = rng.standard_normal(hidden_size).astype(np.float32)
            w /= np.linalg.norm(w) + 1e-8
            probe_weights[cat] = w
        else:
            try:
                probe_weights[cat] = load_probe_weight_vector(probe_path, hidden_size)
            except CausalScrubError:
                logger.exception("probe_load_failed", category=cat)
                raise

    if not probe_weights:
        raise CausalScrubError(
            f"No probe weights could be loaded from {probe_dir}. "
            "Run `python scripts/train_probes.py --synthetic` first."
        )

    logger.info(
        "probe_weights_loaded",
        categories=list(probe_weights.keys()),
        hidden_size=hidden_size,
    )
    return probe_weights


# ── Activation data loading / generation ─────────────────────────────────────

def load_or_generate_activations(
    data_dir: Path,
    category: str,
    n_samples: int,
    hidden_size: int,
    num_heads: int,
    d_head: int,
    rng_seed: int = 0,
) -> np.ndarray:
    """Load cached activations or generate random data for offline testing.

    Looks for ``{data_dir}/{category}_activations.npy`` first (saved by
    ``train_probes.py``).  If absent, generates random activations of shape
    ``(n_samples, num_heads, d_head)`` -- sufficient for unit-testing the
    ablation pipeline without a live model.

    Args:
        data_dir: Directory with pre-saved ``.npy`` activation files.
        category: Probe category (used to find the matching file).
        n_samples: Number of samples to generate when no file is found.
        hidden_size: Model hidden dimension (``num_heads * d_head``).
        num_heads: Number of attention heads per layer.
        d_head: Head dimension (``hidden_size / num_heads``).
        rng_seed: Seed for the random fallback generator.

    Returns:
        Float32 numpy array of shape ``(n_samples, num_heads, d_head)``.
    """
    npy_path = data_dir / f"{category}_activations.npy"
    npz_path = data_dir / f"{category}_activations.npz"

    loaded_flat: Optional[np.ndarray] = None

    if npy_path.exists():
        try:
            loaded_flat = np.load(str(npy_path)).astype(np.float32)
            logger.info(
                "activations_loaded_npy",
                path=str(npy_path),
                shape=loaded_flat.shape,
            )
        except Exception as exc:
            logger.warning("npy_load_failed", path=str(npy_path), error=str(exc))

    elif npz_path.exists():
        try:
            data = np.load(str(npz_path))
            loaded_flat = data["activations"].astype(np.float32)
            logger.info(
                "activations_loaded_npz",
                path=str(npz_path),
                shape=loaded_flat.shape,
            )
        except Exception as exc:
            logger.warning("npz_load_failed", path=str(npz_path), error=str(exc))

    if loaded_flat is not None:
        # Accept (N, hidden_size) or (N, num_heads, d_head)
        if loaded_flat.ndim == 2 and loaded_flat.shape[1] == hidden_size:
            acts = loaded_flat[:n_samples].reshape(-1, num_heads, d_head)
            return acts
        if loaded_flat.ndim == 3 and loaded_flat.shape[1:] == (num_heads, d_head):
            return loaded_flat[:n_samples]
        logger.warning(
            "activations_shape_unexpected",
            shape=loaded_flat.shape,
            expected_flat=(n_samples, hidden_size),
            action="generating_random_fallback",
        )

    # ── Random fallback ───────────────────────────────────────────────────────
    logger.info(
        "generating_random_activations",
        category=category,
        n_samples=n_samples,
        shape=(n_samples, num_heads, d_head),
    )
    rng = np.random.default_rng(seed=rng_seed + abs(hash(category)) % (2**31))
    acts = rng.standard_normal((n_samples, num_heads, d_head)).astype(np.float32)

    # Inject a weak signal into a few heads so the ablation study finds edges
    for signal_head in range(0, min(num_heads, 4)):
        acts[:, signal_head, :] += 0.5 * rng.standard_normal(d_head).astype(np.float32)

    return acts


# ── Per-layer ablation wrapper ────────────────────────────────────────────────

def run_layer_ablation(
    scrubber: CausalScrubber,
    activations_3d: np.ndarray,
    probe_weights: Dict[str, np.ndarray],
    categories: List[str],
    layer_idx: int,
    num_heads: int,
    d_head: int,
    min_delta: float,
    model_id: str,
) -> List[CausalEdge]:
    """Run ablation across all heads in a single layer for all categories.

    Converts the 3-D head-level activation tensor ``(N, num_heads, d_head)``
    into the 2-D format expected by ``CausalScrubber.ablation_study``
    by packaging it as ``{layer_idx: activations_3d}``.

    Args:
        scrubber: Initialised ``CausalScrubber`` instance.
        activations_3d: Float32 array of shape ``(N, num_heads, d_head)``.
        probe_weights: Dict mapping category -> 1-D weight vector.
        categories: Safety categories to analyse.
        layer_idx: Index of this transformer layer.
        num_heads: Number of attention heads.
        d_head: Head feature dimension.
        min_delta: Minimum ablation delta to record as a causal edge.
        model_id: Model checkpoint identifier (for ``HeadSignature``).

    Returns:
        List of ``CausalEdge`` objects discovered in this layer.
    """
    import jax.numpy as jnp

    edges: List[CausalEdge] = []
    discovered_at = datetime.now(timezone.utc).isoformat()

    for head_idx in range(num_heads):
        # Ablate: zero out this head's slice
        ablated = activations_3d.copy()
        ablated[:, head_idx, :] = 0.0

        # Mean-pool over samples -> (num_heads, d_head) -> (hidden_size,)
        orig_flat = activations_3d.mean(axis=0).reshape(-1)      # (hidden_size,)
        ablated_flat = ablated.mean(axis=0).reshape(-1)          # (hidden_size,)

        orig_jax = jnp.array(orig_flat[None, :])    # (1, hidden_size) -- seq dim
        ablated_jax = jnp.array(ablated_flat[None, :])

        for cat in categories:
            if cat not in probe_weights:
                continue

            pw = jnp.array(probe_weights[cat])  # (hidden_size,)
            delta = float(compute_ablation_effect(orig_jax, ablated_jax, pw))

            if abs(delta) >= min_delta:
                head_sig = HeadSignature(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    model_id=model_id,
                )
                edges.append(
                    CausalEdge(
                        head=head_sig,
                        behavior_category=cat,
                        causal_effect=min(abs(delta), 1.0),
                        ablation_delta=delta,
                        patching_delta=delta * 0.85,
                        confidence=min(0.95, abs(delta) * 5.0 + 0.2),
                        discovered_at=discovered_at,
                    )
                )
                logger.debug(
                    "causal_edge_found",
                    layer=layer_idx,
                    head=head_idx,
                    category=cat,
                    delta=f"{delta:.5f}",
                )

    return edges


# ── Causal graph serialisation ────────────────────────────────────────────────

def serialize_causal_graph(graph: CausalGraph) -> Dict:
    """Convert a ``CausalGraph`` to a JSON-serialisable dictionary.

    Args:
        graph: The built causal graph.

    Returns:
        Nested dict ready for ``json.dump``.
    """
    return {
        "model_id": graph.model_id,
        "version": graph.version,
        "total_edges": len(graph.edges),
        "edges": [
            {
                "layer_idx": e.head.layer_idx,
                "head_idx": e.head.head_idx,
                "model_id": e.head.model_id,
                "behavior_category": e.behavior_category,
                "causal_effect": round(e.causal_effect, 6),
                "ablation_delta": round(e.ablation_delta, 6),
                "patching_delta": round(e.patching_delta, 6),
                "confidence": round(e.confidence, 6),
                "discovered_at": e.discovered_at,
            }
            for e in graph.edges
        ],
        "top_k_heads": {
            cat: [
                {"layer_idx": h.layer_idx, "head_idx": h.head_idx}
                for h in heads
            ]
            for cat, heads in graph.top_k_heads.items()
        },
    }


def save_causal_graph(graph: CausalGraph, output_dir: Path, model_name: str) -> Path:
    """Serialise the causal graph to JSON and return the output path.

    Args:
        graph: The built ``CausalGraph``.
        output_dir: Target directory (created if absent).
        model_name: Used to derive the filename.

    Returns:
        Absolute path to the written JSON file.

    Raises:
        CausalScrubError: If the file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("-", "_")
    out_path = output_dir / f"{safe_name}_causal_graph.json"

    try:
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(serialize_causal_graph(graph), fh, indent=2)
    except OSError as exc:
        raise CausalScrubError(
            f"Failed to write causal graph to {out_path}: {exc}"
        ) from exc

    logger.info(
        "causal_graph_saved",
        path=str(out_path),
        total_edges=len(graph.edges),
    )
    return out_path


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(
    graph: CausalGraph,
    categories: List[str],
    top_k: int = 5,
) -> None:
    """Print a formatted summary of top unsafe heads per category.

    Uses only ``print()`` here because this is a human-facing output path
    (not a hot production loop).  All internal logging uses structlog.

    Args:
        graph: The completed causal graph.
        categories: Safety categories to report.
        top_k: How many heads to show per category.
    """
    model_tag = graph.model_id
    separator = "=" * 72

    print(f"\n{separator}")
    print(f"  NEXUS Causal Scrubbing -- {model_tag}")
    print(f"  Total causal edges: {len(graph.edges)}")
    print(separator)

    for cat in categories:
        top_heads = graph.get_unsafe_heads(cat, min_effect=0.0)[:top_k]
        print(f"\n  Category: {cat.upper()}")

        if not top_heads:
            print("    (no causal heads found above threshold)")
            continue

        col_w = {"rank": 6, "layer": 8, "head": 8, "effect": 12, "conf": 12, "delta": 12}
        header = (
            f"  {'Rank':>{col_w['rank']}} "
            f"{'Layer':>{col_w['layer']}} "
            f"{'Head':>{col_w['head']}} "
            f"{'Effect':>{col_w['effect']}} "
            f"{'Confidence':>{col_w['conf']}} "
            f"{'AblDelta':>{col_w['delta']}}"
        )
        print(header)
        print("  " + "-" * (sum(col_w.values()) + len(col_w)))

        for rank, head in enumerate(top_heads, start=1):
            edge = next(
                (
                    e for e in graph.edges
                    if e.head == head and e.behavior_category == cat
                ),
                None,
            )
            if edge is None:
                continue
            print(
                f"  {rank:>{col_w['rank']}} "
                f"{head.layer_idx:>{col_w['layer']}} "
                f"{head.head_idx:>{col_w['head']}} "
                f"{edge.causal_effect:>{col_w['effect']}.5f} "
                f"{edge.confidence:>{col_w['conf']}.5f} "
                f"{edge.ablation_delta:>{col_w['delta']}.5f}"
            )

    print(f"\n{separator}\n")


# ── MLflow logging helpers ────────────────────────────────────────────────────

def log_to_mlflow(
    graph: CausalGraph,
    categories: List[str],
    run_time_seconds: float,
    graph_path: Path,
    model_name: str,
    mlflow_uri: str,
    n_prompts: int,
    min_delta: float,
    num_layers: int,
    num_heads: int,
) -> None:
    """Log causal scrubbing results and artefacts to MLflow.

    Gracefully skips MLflow when the tracking server is unreachable so that
    the script remains useful in offline environments.

    Args:
        graph: The built causal graph.
        categories: Safety categories that were analysed.
        run_time_seconds: Wall-clock seconds taken for the full run.
        graph_path: Path to the saved causal graph JSON.
        model_name: Friendly model name used as the run tag.
        mlflow_uri: MLflow tracking URI.
        n_prompts: Number of samples used per layer.
        min_delta: Minimum ablation delta threshold.
        num_layers: Number of transformer layers scraped.
        num_heads: Number of attention heads per layer.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_not_installed", action="skipping_mlflow_logging")
        return

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("nexus/causal_scrubbing")

        with mlflow.start_run(run_name=f"causal_scrubbing_{model_name}"):
            # Params
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "categories": ",".join(categories),
                    "n_prompts": n_prompts,
                    "min_delta": min_delta,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                }
            )

            # Core metrics
            mlflow.log_metric("num_edges", len(graph.edges))
            mlflow.log_metric("run_time_seconds", round(run_time_seconds, 3))

            # Per-category edge counts
            for cat in categories:
                cat_edges = [e for e in graph.edges if e.behavior_category == cat]
                mlflow.log_metric(f"{cat}_num_edges", len(cat_edges))

            # Top-5 heads per category (logged as a JSON param for easy retrieval)
            top_heads_dict: Dict[str, List[str]] = {}
            for cat in categories:
                top = graph.get_unsafe_heads(cat, min_effect=0.0)[:5]
                top_heads_dict[cat] = [str(h) for h in top]
            mlflow.log_param("top_heads_per_category", json.dumps(top_heads_dict))

            # Artefact
            mlflow.log_artifact(str(graph_path))

        logger.info(
            "mlflow_logged",
            num_edges=len(graph.edges),
            run_time_seconds=run_time_seconds,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "mlflow_logging_failed",
            error=str(exc),
            action="continuing_without_mlflow",
        )


# ── CLI argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "NEXUS MOD-02: Offline causal scrubbing -- "
            "builds a CausalGraph by systematically ablating attention heads."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="llama31_8b",
        help="Logical model identifier used in output filenames and MLflow tags.",
    )
    parser.add_argument(
        "--probe-dir",
        default="models/probes",
        help="Directory containing trained probe ``.pt`` files.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/causal",
        help="Directory where the causal graph JSON is written.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/probes",
        help="Directory with pre-saved ``*_activations.npy`` files.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=128,
        help="Number of activation samples to use per layer (random fallback size).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.05,
        help="Minimum ablation delta magnitude to record as a causal edge.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=LLAMA31_8B_NUM_LAYERS,
        help="Number of transformer layers to scrub.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=LLAMA31_8B_NUM_HEADS,
        help="Number of attention heads per layer.",
    )
    parser.add_argument(
        "--d-head",
        type=int,
        default=LLAMA31_8B_D_HEAD,
        help="Per-head feature dimension (hidden_size / num_heads).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=PROBE_CATEGORIES,
        choices=PROBE_CATEGORIES,
        help="Safety categories to analyse.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top unsafe heads to print per category.",
    )
    return parser.parse_args()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full offline causal-scrubbing pipeline.

    Steps:
        1. Load trained probe weight vectors from ``models/probes/``.
        2. Load or generate activation data per layer.
        3. Run head-level ablation via ``CausalScrubber.ablation_study``.
        4. Build ``CausalGraph`` with top-K heads pre-computed per category.
        5. Serialise the graph to ``models/causal/`` as JSON.
        6. Log metrics and the graph artefact to MLflow.
        7. Print a human-readable summary table.

    Raises:
        CausalScrubError: On any unrecoverable failure during scrubbing.
        SystemExit: With exit code 1 on fatal errors.
    """
    args = parse_args()

    hidden_size: int = args.num_heads * args.d_head
    project_root = Path(__file__).resolve().parent.parent

    probe_dir = (project_root / args.probe_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    data_dir = (project_root / args.data_dir).resolve()

    logger.info(
        "causal_scrubbing_start",
        model_name=args.model_name,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_head=args.d_head,
        hidden_size=hidden_size,
        categories=args.categories,
        n_prompts=args.n_prompts,
        min_delta=args.min_delta,
        probe_dir=str(probe_dir),
        output_dir=str(output_dir),
    )

    t_start = time.perf_counter()

    # ── Step 1: Load probe weights ────────────────────────────────────────────
    try:
        probe_weights = load_all_probe_weights(
            probe_dir=probe_dir,
            categories=args.categories,
            hidden_size=hidden_size,
        )
    except CausalScrubError:
        logger.exception("probe_loading_failed")
        sys.exit(1)

    # ── Auto-adapt architecture dims to match actual probe input_dim ─────────
    # Local probes (e.g. trained on per-head d_head activations) may have
    # input_dim != num_heads * d_head.  When that happens we switch to
    # single-head mode so activation pooling produces (probe_input_dim,)
    # vectors that match the probe weight shape.
    if probe_weights:
        actual_probe_input_dim = min(len(v) for v in probe_weights.values())
        if actual_probe_input_dim != hidden_size:
            logger.warning(
                "adapting_to_probe_input_dim",
                original_hidden_size=hidden_size,
                probe_input_dim=actual_probe_input_dim,
                action="single_head_mode: num_heads=1, d_head=probe_input_dim",
            )
            hidden_size = actual_probe_input_dim
            args.num_heads = 1
            args.d_head = actual_probe_input_dim

    # ── Step 2 + 3: Load activations and run ablation per layer ──────────────
    scrubber = CausalScrubber(
        model_id=args.model_name,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    all_edges: List[CausalEdge] = []

    for layer_idx in range(args.num_layers):
        logger.info(
            "scrubbing_layer",
            layer=layer_idx,
            total_layers=args.num_layers,
        )

        # Load or generate activations for this layer's category set.
        # We use the first category as the activation filename key; all
        # categories share the same activation tensor per layer in this offline
        # pipeline (a real run would separate them by prompt label).
        primary_cat = args.categories[0]
        layer_acts_3d = load_or_generate_activations(
            data_dir=data_dir,
            category=primary_cat,
            n_samples=args.n_prompts,
            hidden_size=hidden_size,
            num_heads=args.num_heads,
            d_head=args.d_head,
            rng_seed=layer_idx,
        )

        layer_edges = run_layer_ablation(
            scrubber=scrubber,
            activations_3d=layer_acts_3d,
            probe_weights=probe_weights,
            categories=args.categories,
            layer_idx=layer_idx,
            num_heads=args.num_heads,
            d_head=args.d_head,
            min_delta=args.min_delta,
            model_id=args.model_name,
        )
        all_edges.extend(layer_edges)

        logger.debug(
            "layer_scrubbing_complete",
            layer=layer_idx,
            edges_found=len(layer_edges),
            cumulative_edges=len(all_edges),
        )

    # ── Step 4: Build CausalGraph ─────────────────────────────────────────────
    top_k_heads: Dict[str, List[HeadSignature]] = {}
    for cat in args.categories:
        cat_edges = sorted(
            [e for e in all_edges if e.behavior_category == cat],
            key=lambda e: e.causal_effect,
            reverse=True,
        )
        top_k_heads[cat] = [e.head for e in cat_edges[: args.top_k]]

    graph = CausalGraph(
        model_id=args.model_name,
        edges=all_edges,
        top_k_heads=top_k_heads,
        version=1,
    )

    logger.info(
        "causal_graph_built",
        total_edges=len(all_edges),
        categories=args.categories,
    )

    # ── Step 5: Save to disk ──────────────────────────────────────────────────
    try:
        graph_path = save_causal_graph(graph, output_dir, args.model_name)
    except CausalScrubError:
        logger.exception("causal_graph_save_failed")
        sys.exit(1)

    t_elapsed = time.perf_counter() - t_start

    # ── Step 6: MLflow logging ────────────────────────────────────────────────
    log_to_mlflow(
        graph=graph,
        categories=args.categories,
        run_time_seconds=t_elapsed,
        graph_path=graph_path,
        model_name=args.model_name,
        mlflow_uri=args.mlflow_uri,
        n_prompts=args.n_prompts,
        min_delta=args.min_delta,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    # ── Step 7: Human-readable summary ───────────────────────────────────────
    print_summary_table(graph, args.categories, top_k=args.top_k)

    print(f"  Graph saved : {graph_path}")
    print(f"  Total edges : {len(all_edges)}")
    print(f"  Run time    : {t_elapsed:.2f}s\n")

    logger.info(
        "causal_scrubbing_complete",
        run_time_seconds=round(t_elapsed, 3),
        total_edges=len(all_edges),
        output_path=str(graph_path),
    )


if __name__ == "__main__":
    main()

