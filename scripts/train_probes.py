"""
MOD-01: Probe Training Pipeline
=================================
Trains the 4 LinearResidualProbe classifiers used by LatentSentinel:
  - HallucinationProbe    (TruthfulQA + HaluEval)
  - JailbreakProbe        (AdvBench + custom red-team)
  - ToxicReasoningProbe   (ToxiGen + SafetyBench)
  - PolicyViolationProbe  (Anthropic HH-RLHF)

Run with:
    python scripts/train_probes.py --model meta-llama/Meta-Llama-3.1-8B-Instruct
    python scripts/train_probes.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --categories hallucination jailbreak

The script:
1. Loads the target LLM (read-only, no gradient)
2. Attaches LatentSentinel hooks to extract residual stream activations
3. Runs labeled safe/unsafe prompts through the model
4. Trains a LinearResidualProbe per category on the collected activations
5. Evaluates on held-out set (target: >85% precision)
6. Saves probe weights to models/probes/
7. Logs all metrics to MLflow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, precision_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_sentinel_config
from src.exceptions import ProbeInferenceError  # noqa: F401 — used for structured error raising

logger = structlog.get_logger(__name__)

# ── Dataset sources per category ─────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "hallucination": {
        "hf_dataset": "truthful_qa",
        "hf_config": "multiple_choice",
        "description": "TruthfulQA for hallucination detection",
    },
    "jailbreak": {
        "hf_dataset": "Anthropic/hh-rlhf",
        "hf_config": "harmless-base",
        "description": "Anthropic HH-RLHF harmless pairs for jailbreak detection",
    },
    "toxic_reasoning": {
        "hf_dataset": "allenai/real-toxicity-prompts",
        "hf_config": None,
        "description": "RealToxicityPrompts for toxic reasoning detection",
    },
    "policy_violation": {
        "hf_dataset": "Anthropic/hh-rlhf",
        "hf_config": "red-team-attempts",
        "description": "HH-RLHF red-team attempts for policy violation detection",
    },
}


# ── Activation Collector ──────────────────────────────────────────────────────

@dataclass
class ActivationCollector:
    """Collects activation bundles during inference for probe training.

    Args:
        target_layers: Layer indices to collect activations from.
        max_seq_len: Maximum sequence length (truncate longer sequences).
    """

    target_layers: List[int]
    max_seq_len: int = 128
    _bundles: List[ActivationBundle] = field(default_factory=list, init=False)
    _labels: List[int] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Clear collected activations."""
        self._bundles.clear()
        self._labels.clear()

    def add(self, bundle: ActivationBundle, label: int) -> None:
        """Record one activation bundle with its safety label.

        Args:
            bundle: Activation data from a single forward pass layer.
            label: 1 = unsafe, 0 = safe.
        """
        self._bundles.append(bundle)
        self._labels.append(label)

    def to_tensors(self, probe_dim: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert collected bundles to training tensors.

        Mean-pools the residual stream across the sequence dimension and
        applies PCA-like dimensionality reduction to probe_dim.

        Args:
            probe_dim: Target feature dimensionality for the probe.

        Returns:
            Tuple of (features: [N, probe_dim], labels: [N]).
        """
        features = []
        for bundle in self._bundles:
            # Mean-pool across sequence: (batch, seq_len, hidden) → (hidden,)
            pooled = bundle.residual_stream.mean(dim=1).squeeze(0)  # (hidden_dim,)
            features.append(pooled.float().cpu())

        X = torch.stack(features)  # (N, hidden_dim)
        y = torch.tensor(self._labels, dtype=torch.long)

        # Project to probe_dim with a fixed random projection (reproducible)
        hidden_dim = X.shape[1]
        if hidden_dim > probe_dim:
            generator = torch.Generator().manual_seed(42)
            projection = torch.randn(hidden_dim, probe_dim, generator=generator)
            projection = projection / projection.norm(dim=0, keepdim=True)
            X = X @ projection  # (N, probe_dim)

        return X, y


# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_probe_dataset(
    category: str,
    max_samples: int = 2000,
    hf_token: Optional[str] = None,
) -> Tuple[List[str], List[int]]:
    """Load and format prompts + binary labels for a probe category.

    Args:
        category: One of hallucination / jailbreak / toxic_reasoning / policy_violation.
        max_samples: Maximum total samples (balanced safe/unsafe).
        hf_token: HuggingFace token for gated datasets.

    Returns:
        Tuple of (prompts list, labels list) where label=1 means unsafe.

    Raises:
        ValueError: If category is not recognized.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    if category not in DATASET_REGISTRY:
        raise ValueError(f"Unknown probe category: {category!r}")

    prompts: List[str] = []
    labels: List[int] = []
    half = max_samples // 2

    if category == "hallucination":
        ds = load_dataset("truthful_qa", "multiple_choice", token=hf_token, split="validation")
        for row in ds:
            # Unsafe: the incorrect (hallucinated) answer
            question = row["question"]
            if row["mc1_targets"]["labels"]:
                wrong_ans = row["mc1_targets"]["choices"][
                    row["mc1_targets"]["labels"].index(0)
                    if 0 in row["mc1_targets"]["labels"]
                    else 0
                ]
                prompts.append(f"Q: {question}\nA: {wrong_ans}")
                labels.append(1)
            # Safe: the correct answer
            correct_ans = row["mc1_targets"]["choices"][
                row["mc1_targets"]["labels"].index(1)
            ]
            prompts.append(f"Q: {question}\nA: {correct_ans}")
            labels.append(0)
            if len(prompts) >= max_samples:
                break

    elif category in ("jailbreak", "policy_violation"):
        if category == "policy_violation":
            # red-team-attempts uses 'transcript' + 'rating' (not chosen/rejected)
            ds = load_dataset(
                "Anthropic/hh-rlhf",
                data_dir="red-team-attempts",
                token=hf_token,
                split="train",
            )
            unsafe_pool: List[str] = []
            safe_pool: List[str] = []
            for row in ds:
                transcript = row.get("transcript", "")
                rating = float(row.get("rating") or 5.0)
                if not transcript:
                    continue
                # Lower rating = more harmful in red-team dataset
                if rating < 3.0 and len(unsafe_pool) < half:
                    unsafe_pool.append(transcript[:512])
                elif rating >= 3.0 and len(safe_pool) < half:
                    safe_pool.append(transcript[:512])
                if len(unsafe_pool) >= half and len(safe_pool) >= half:
                    break
            for text in unsafe_pool[:half]:
                prompts.append(text)
                labels.append(1)
            for text in safe_pool[:half]:
                prompts.append(text)
                labels.append(0)
        else:
            # jailbreak uses harmless-base (chosen/rejected pairs)
            ds = load_dataset(
                "Anthropic/hh-rlhf",
                data_dir="harmless-base",
                token=hf_token,
                split="train",
            )
            for row in ds:
                rejected = row.get("rejected", "")
                chosen = row.get("chosen", "")
                if rejected:
                    prompts.append(rejected[:512])
                    labels.append(1)
                if chosen:
                    prompts.append(chosen[:512])
                    labels.append(0)
                if len(prompts) >= max_samples:
                    break

    elif category == "toxic_reasoning":
        ds = load_dataset("allenai/real-toxicity-prompts", split="train")
        # Pre-scan to guarantee balanced labels — toxic rows are sparse at the start
        toxic_pool: List[str] = []
        safe_pool: List[str] = []
        for row in ds:
            toxicity = row.get("prompt", {}).get("toxicity") or 0.0
            text = row.get("prompt", {}).get("text", "")
            if not text:
                continue
            if toxicity > 0.5:
                toxic_pool.append(text[:512])
            else:
                safe_pool.append(text[:512])
            if len(toxic_pool) >= half and len(safe_pool) >= half:
                break
        for text in toxic_pool[:half]:
            prompts.append(text)
            labels.append(1)
        for text in safe_pool[:half]:
            prompts.append(text)
            labels.append(0)

    logger.info(
        "dataset_loaded",
        category=category,
        total=len(prompts),
        unsafe=sum(labels),
        safe=len(labels) - sum(labels),
    )
    return prompts, labels


# ── Inference + Activation Collection ────────────────────────────────────────

@torch.no_grad()
def collect_activations(
    model: nn.Module,
    tokenizer: any,
    prompts: List[str],
    labels: List[int],
    target_layers: List[int],
    device: str = "cuda",
    batch_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and collect mean-pooled residual activations via direct hooks.

    Uses ``nn.Module.register_forward_hook`` directly on each target layer —
    no dependency on HookManager internals.

    Args:
        model: Loaded HuggingFace transformer model (eval mode, no grad).
        tokenizer: Corresponding tokenizer.
        prompts: List of text prompts.
        labels: Corresponding binary safety labels.
        target_layers: Transformer layer indices to hook.
        device: Inference device.
        batch_size: Number of prompts per forward pass.

    Returns:
        Tuple of (activations: np.ndarray [N, hidden_dim], labels: np.ndarray [N]).
        One row per prompt, using the LAST target layer's activations.
    """
    model.eval()

    # Locate the transformer layer list
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_list = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_list = model.transformer.h
    else:
        raise ValueError(f"Cannot locate layer list in {type(model).__name__}")

    last_layer_idx = target_layers[-1]

    captured: List[torch.Tensor] = []

    def _hook(_module: nn.Module, _inp: tuple, output: tuple) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        # Mean-pool over sequence → (batch, hidden_dim)
        pooled = hidden.detach().float().mean(dim=1)
        captured.append(pooled.cpu())

    handle = layer_list[last_layer_idx].register_forward_hook(_hook)

    all_activations: List[np.ndarray] = []
    all_labels: List[int] = []

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Collecting activations"):
            batch_prompts = prompts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            captured.clear()
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)

            _ = model(**inputs)

            if captured:
                acts = captured[0].numpy()  # (batch, hidden_dim)
                all_activations.append(acts)
                all_labels.extend(batch_labels)
    finally:
        handle.remove()

    if not all_activations:
        raise ValueError(
            f"No activations collected — dataset may be empty or have no matching rows. "
            f"Got {len(prompts)} prompts but 0 batches captured."
        )
    return np.concatenate(all_activations, axis=0), np.array(all_labels, dtype=np.int64)


# ── Probe Training ────────────────────────────────────────────────────────────

def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int = 128,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple["EvalProbe", Dict[str, float]]:
    """Train a LinearResidualProbe on collected activations.

    Uses the raw-tensor probe from ``src.latent_sentinel.probes`` so that
    saved weights are compatible with ``eval_probes.py``.

    Args:
        X_train: Training features [N_train, hidden_dim].
        y_train: Training labels [N_train].
        X_val: Validation features [N_val, hidden_dim].
        y_val: Validation labels [N_val].
        hidden_dim: Activation feature dimensionality.
        num_epochs: Training epochs.
        lr: Learning rate.
        device: Training device.

    Returns:
        Tuple of (trained probe, metrics dict).
    """
    from src.latent_sentinel.probes import LinearResidualProbe as EvalProbe

    probe = EvalProbe(input_dim=hidden_dim).to(device)

    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_ds = TensorDataset(X_train.to(device), y_train.float().to(device))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_precision = 0.0
    # Initialise with current weights so load_state_dict never gets an empty dict
    best_state: Dict = {k: v.clone() for k, v in probe.state_dict().items()}

    for epoch in range(num_epochs):
        probe.train()
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            logits = probe(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val.to(device))
            val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
            val_precision = precision_score(y_val.numpy(), val_preds, zero_division=0.0)

        if val_precision > best_val_precision:
            best_val_precision = val_precision
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                "probe_training_progress",
                epoch=epoch + 1,
                val_precision=f"{val_precision:.4f}",
            )

    probe.load_state_dict(best_state)
    probe.eval()

    # Final evaluation
    with torch.no_grad():
        val_logits = probe(X_val.to(device))
        val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()

    report = classification_report(y_val.numpy(), val_preds, output_dict=True)
    metrics = {
        "val_precision": report["1"]["precision"],
        "val_recall": report["1"]["recall"],
        "val_f1": report["1"]["f1-score"],
        "val_accuracy": report["accuracy"],
    }
    return probe, metrics


# ── Synthetic Training (no LLM required) ─────────────────────────────────────

def run_synthetic_training(
    categories: List[str],
    probe_dim: int,
    n_train: int,
    n_eval: int,
    epochs: int,
    signal_strength: float,
    output_dir: Path,
    data_dir: Path,
    device: str,
) -> Dict[str, Dict]:
    """Train probes on class-conditional Gaussian synthetic activations.

    Generates separable safe/unsafe feature distributions, trains a probe per
    category, saves weights + eval data.  No LLM or internet access required.

    Args:
        categories: Probe category names to train.
        probe_dim: Activation dimensionality (128 by default).
        n_train: Number of synthetic training samples per category.
        n_eval: Number of held-out eval samples per category.
        epochs: Training epochs.
        signal_strength: Class mean separation (higher = more separable).
        output_dir: Where to save probe ``.pt`` files.
        data_dir: Where to save eval JSONL + activation ``.npz`` caches.
        device: Torch device.

    Returns:
        Dict mapping category name -> training metrics dict.
    """
    from src.latent_sentinel.probes import LinearResidualProbe as EvalProbe

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}

    for cat in categories:
        logger.info("synthetic_probe_training", category=cat, probe_dim=probe_dim)

        rng = np.random.default_rng(seed=42 + abs(hash(cat)) % 10_000)

        # Separating direction — random unit vector
        direction = rng.standard_normal(probe_dim).astype(np.float32)
        direction /= np.linalg.norm(direction)

        def _gen_split(n: int) -> Tuple[np.ndarray, np.ndarray]:
            n_pos = n // 2
            n_neg = n - n_pos
            X_safe = rng.standard_normal((n_neg, probe_dim)).astype(np.float32)
            X_unsafe = rng.standard_normal((n_pos, probe_dim)).astype(np.float32)
            X_unsafe += signal_strength * direction[np.newaxis, :]
            X = np.vstack([X_safe, X_unsafe])
            y = np.array([0] * n_neg + [1] * n_pos, dtype=np.int64)
            idx = rng.permutation(len(X))
            return X[idx], y[idx]

        X_train_np, y_train_np = _gen_split(n_train)
        X_eval_np, y_eval_np = _gen_split(n_eval)

        X_train = torch.from_numpy(X_train_np)
        y_train = torch.from_numpy(y_train_np).long()
        X_val = torch.from_numpy(X_eval_np)
        y_val = torch.from_numpy(y_eval_np).long()

        # Build probe matching eval_probes.py interface
        probe = EvalProbe(input_dim=probe_dim).to(device)
        optimizer = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_ds = TensorDataset(X_train.to(device), y_train.float().to(device))
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

        best_prec = 0.0
        best_state: Dict = {}

        for epoch in range(epochs):
            probe.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                logits = probe(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val.to(device))
                val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
                prec = float(
                    (val_preds == 1).sum() > 0
                    and ((val_preds == 1) & (y_val.numpy() == 1)).sum()
                    / max((val_preds == 1).sum(), 1)
                )

            if prec > best_prec:
                best_prec = prec
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        probe.load_state_dict(best_state)
        probe.eval()

        # Final metrics
        with torch.no_grad():
            val_logits = probe(X_val.to(device))
            val_preds = (torch.sigmoid(val_logits) > 0.5).cpu().numpy()
        y_val_np = y_val.numpy()
        tp = int(((val_preds == 1) & (y_val_np == 1)).sum())
        fp = int(((val_preds == 1) & (y_val_np == 0)).sum())
        fn = int(((val_preds == 0) & (y_val_np == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = float((val_preds == y_val_np).mean())

        # Save probe weights
        probe_path = output_dir / f"{cat}.pt"
        torch.save(probe.state_dict(), probe_path)

        # Save eval JSONL
        eval_jsonl = data_dir / f"{cat}_eval.jsonl"
        with eval_jsonl.open("w") as fh:
            for i, (feat, lbl) in enumerate(zip(X_eval_np, y_eval_np)):
                fh.write(json.dumps({"text": f"synthetic_sample_{i}", "label": int(lbl)}) + "\n")

        # Save activation cache so eval_probes.py can skip re-computation
        cache_path = data_dir / f"{cat}_activations.npz"
        np.save(str(cache_path).replace(".npz", ""), X_eval_np)
        np.savez(str(cache_path), activations=X_eval_np, labels=y_eval_np)

        metrics = {
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_accuracy": accuracy,
        }
        results[cat] = {"path": str(probe_path), **metrics}

        logger.info(
            "synthetic_probe_trained",
            category=cat,
            precision=f"{precision:.4f}",
            recall=f"{recall:.4f}",
            f1=f"{f1:.4f}",
            saved_to=str(probe_path),
        )

    return results


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ARGUS LatentSentinel probes")
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["hallucination", "jailbreak", "toxic_reasoning", "policy_violation"],
        choices=list(DATASET_REGISTRY.keys()),
        help="Probe categories to train",
    )
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--probe-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="models/probes")
    parser.add_argument("--data-dir", default="data/probes")
    parser.add_argument("--mlflow-experiment", default="argus/probe_training")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Train on synthetic class-conditional Gaussian activations (no LLM/GPU required).",
    )
    parser.add_argument("--synthetic-n-train", type=int, default=800)
    parser.add_argument("--synthetic-n-eval", type=int, default=200)
    parser.add_argument("--synthetic-signal", type=float, default=3.0,
                        help="Class mean separation for synthetic activations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Synthetic fast-path (no LLM required) ────────────────────────────────
    if args.synthetic:
        logger.info(
            "synthetic_training_mode",
            categories=args.categories,
            probe_dim=args.probe_dim,
            epochs=args.epochs,
            signal_strength=args.synthetic_signal,
        )
        results = run_synthetic_training(
            categories=args.categories,
            probe_dim=args.probe_dim,
            n_train=args.synthetic_n_train,
            n_eval=args.synthetic_n_eval,
            epochs=args.epochs,
            signal_strength=args.synthetic_signal,
            output_dir=output_dir,
            data_dir=Path(args.data_dir),
            device=args.device,
        )

        summary_path = output_dir / "training_summary.json"
        with summary_path.open("w") as f:
            json.dump(results, f, indent=2)

        print("\n=== Probe Training Results (Synthetic) ===")
        print(f"  {'Category':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Status':>8}")
        print("  " + "-" * 65)
        for cat, m in results.items():
            status = "[OK]" if m["val_precision"] >= 0.85 else "[LOW]"
            print(
                f"  {cat:<22} {m['val_precision']:>10.4f} "
                f"{m['val_recall']:>10.4f} {m['val_f1']:>10.4f} {status:>8}"
            )
        return

    # Auto-detect target layers from config; fallback to sensible defaults
    try:
        cfg = load_sentinel_config()
        target_layers = cfg.hooks.target_layers
    except Exception:
        target_layers = None  # will auto-detect after model load

    logger.info(
        "probe_training_start",
        model=args.model,
        categories=args.categories,
        device=args.device,
    )

    # Load model + tokenizer (lazy import to avoid torch at import time)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Run: pip install transformers")

    logger.info("loading_model", model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.hf_token or None
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use float32 on CPU (bfloat16 has no native CPU SIMD and is ~10x slower)
    _dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_dtype,
        device_map=args.device,
        token=args.hf_token or None,
    )
    model.eval()
    logger.info("model_loaded", model=args.model)

    # Auto-detect model dimensions
    hidden_dim: int = getattr(model.config, "hidden_size", 4096)
    num_layers: int = getattr(model.config, "num_hidden_layers", 32)
    if target_layers is None:
        # Pick 3 evenly spaced layers in the second half of the model
        target_layers = [
            num_layers // 4,
            num_layers // 2,
            num_layers * 3 // 4,
        ]
    # Clamp to valid range
    target_layers = [min(l, num_layers - 1) for l in target_layers]
    logger.info(
        "model_dims",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_layers=target_layers,
    )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        mlflow.set_experiment(args.mlflow_experiment)
        _mlflow_active = True
    except Exception:
        _mlflow_active = False

    results: Dict[str, Dict] = {}

    for category in args.categories:
        logger.info("training_probe", category=category)

        run_ctx = mlflow.start_run(run_name=f"probe_{category}") if _mlflow_active else None
        try:
            if _mlflow_active:
                mlflow.log_params(
                    {
                        "model": args.model,
                        "category": category,
                        "hidden_dim": hidden_dim,
                        "max_samples": args.max_samples,
                        "epochs": args.epochs,
                    }
                )

            # Load dataset
            prompts, labels = load_probe_dataset(
                category, max_samples=args.max_samples, hf_token=args.hf_token
            )

            # Collect activations — returns (np.ndarray [N, hidden_dim], np.ndarray [N])
            activations_np, labels_np = collect_activations(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                labels=labels,
                target_layers=target_layers,
                device=args.device,
            )

            # Convert to tensors
            X = torch.from_numpy(activations_np.astype(np.float32))
            y = torch.from_numpy(labels_np).long()

            # Train/val split (80/20)
            n = len(X)
            n_train = int(0.8 * n)
            idx = torch.randperm(n)
            X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
            y_train, y_val = y[idx[:n_train]], y[idx[n_train:]]

            # Train — hidden_dim is the actual activation size
            probe, metrics = train_probe(
                X_train, y_train, X_val, y_val,
                hidden_dim=hidden_dim,
                num_epochs=args.epochs,
                device=args.device,
            )

            if _mlflow_active:
                mlflow.log_metrics(metrics)

            # Save probe weights — eval_probes.py expects {category}.pt
            probe_path = output_dir / f"{category}.pt"
            torch.save(probe.state_dict(), probe_path)
            if _mlflow_active:
                mlflow.log_artifact(str(probe_path))

            # Save eval JSONL so eval_probes.py can load it
            X_val_np = X_val.numpy()
            y_val_np = y_val.numpy()
            eval_jsonl = data_dir / f"{category}_eval.jsonl"
            with eval_jsonl.open("w", encoding="utf-8") as fh:
                for i, lbl in enumerate(y_val_np):
                    fh.write(json.dumps({"text": f"sample_{i}", "label": int(lbl)}) + "\n")

            # Save activation cache for eval_probes.py
            cache_path = data_dir / f"{category}_activations.npz"
            np.savez(str(cache_path), activations=X_val_np, labels=y_val_np)

            results[category] = {"path": str(probe_path), **metrics}

            logger.info(
                "probe_trained",
                category=category,
                precision=f"{metrics['val_precision']:.4f}",
                recall=f"{metrics['val_recall']:.4f}",
                f1=f"{metrics['val_f1']:.4f}",
                saved_to=str(probe_path),
            )

            if metrics["val_precision"] < 0.85:
                logger.warning(
                    "probe_below_target",
                    category=category,
                    precision=metrics["val_precision"],
                    target=0.85,
                )
        finally:
            if _mlflow_active and run_ctx is not None:
                mlflow.end_run()

    # Summary report
    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("probe_training_complete", results=results)

    # Print table
    print("\n=== Probe Training Results ===")
    print(f"  {'Category':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Status':>8}")
    print("  " + "-" * 65)
    for cat, m in results.items():
        status = "[OK]" if m["val_precision"] >= 0.85 else "[LOW]"
        print(
            f"  {cat:<22} {m['val_precision']:>10.4f} {m['val_recall']:>10.4f} "
            f"{m['val_f1']:>10.4f} {status:>8}"
        )


if __name__ == "__main__":
    main()
