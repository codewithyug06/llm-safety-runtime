"""
MOD-02: Causal Intervention Verification Script
================================================
Loads trained probes + a CausalGraph (from a JSON file if it exists, or builds a
synthetic one for testing) and verifies that soft interventions via
CausalInterventionEngine:

  1. Return InterventionResult for unsafe signals.
  2. Return None for safe signals.
  3. Add <5ms overhead per intervention (p95 over 100 calls).

MLflow run name: "intervention_verification"

Run with:
    python scripts/verify_interventions.py
    python scripts/verify_interventions.py \\
        --causal-graph-path models/causal/llama31_8b_causal_graph.json \\
        --n-samples 20 --threshold 0.65 --assert-delta-min 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal_engine.intervention import (
    CausalEdge,
    CausalGraph,
    CausalInterventionEngine,
    HeadSignature,
    InterventionResult,
    InterventionSpec,
)
from src.exceptions import InterventionError

logger = structlog.get_logger(__name__)

# ── Synthetic graph construction ──────────────────────────────────────────────

#: Categories used throughout the verification harness.
CATEGORIES: List[str] = ["jailbreak", "hallucination", "toxic_reasoning", "policy_violation"]

#: Random-state seed for reproducibility.
_RNG = np.random.default_rng(seed=42)


def _make_synthetic_causal_graph(model_id: str = "llama-3.1-8b-stub") -> CausalGraph:
    """Build a realistic synthetic CausalGraph for offline testing.

    Generates 3–4 CausalEdge objects per safety category, with causal_effect
    values uniformly sampled from [0.3, 0.8] to simulate real ablation results.

    Args:
        model_id: Identifier to embed in every HeadSignature.

    Returns:
        CausalGraph with populated edges and top_k_heads.
    """
    edges: List[CausalEdge] = []
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    # Layer/head assignments — spread across early, mid, and late layers
    _head_pool: List[Tuple[int, int]] = [
        (4, 2), (4, 7), (8, 0), (8, 5), (8, 11),
        (12, 3), (12, 8), (16, 1), (16, 6), (20, 4),
        (24, 9), (24, 15), (28, 2), (28, 7),
    ]
    pool_index = 0

    for category in CATEGORIES:
        n_heads = _RNG.integers(3, 5)  # 3 or 4 heads per category
        for _ in range(n_heads):
            layer_idx, head_idx = _head_pool[pool_index % len(_head_pool)]
            pool_index += 1

            causal_effect = float(_RNG.uniform(0.3, 0.8))
            ablation_delta = causal_effect * float(_RNG.uniform(0.6, 1.0))
            patching_delta = ablation_delta * float(_RNG.uniform(0.5, 0.9))
            confidence = float(_RNG.uniform(0.7, 1.0))

            head = HeadSignature(
                layer_idx=layer_idx,
                head_idx=head_idx,
                model_id=model_id,
            )
            edges.append(CausalEdge(
                head=head,
                behavior_category=category,
                causal_effect=causal_effect,
                ablation_delta=ablation_delta,
                patching_delta=patching_delta,
                confidence=confidence,
                discovered_at=now_iso,
            ))

    # Build top_k_heads dict
    top_k: Dict[str, List[HeadSignature]] = {}
    for category in CATEGORIES:
        cat_edges = sorted(
            [e for e in edges if e.behavior_category == category],
            key=lambda e: e.causal_effect,
            reverse=True,
        )
        top_k[category] = [e.head for e in cat_edges[:3]]

    graph = CausalGraph(
        model_id=model_id,
        edges=edges,
        top_k_heads=top_k,
        version=1,
    )
    logger.info(
        "synthetic_causal_graph_built",
        model_id=model_id,
        total_edges=len(edges),
        categories=CATEGORIES,
    )
    return graph


def _load_causal_graph_from_json(path: Path) -> CausalGraph:
    """Deserialise a CausalGraph previously saved as JSON.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        Re-hydrated CausalGraph.

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If required JSON fields are missing.
    """
    logger.info("loading_causal_graph_from_json", path=str(path))
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    edges: List[CausalEdge] = []
    for raw in data.get("edges", []):
        # Support both nested {"head": {...}} and flat {"layer_idx": ..., "head_idx": ...}
        if "head" in raw:
            head_raw = raw["head"]
            head = HeadSignature(
                layer_idx=int(head_raw["layer_idx"]),
                head_idx=int(head_raw["head_idx"]),
                model_id=str(head_raw["model_id"]),
            )
        else:
            head = HeadSignature(
                layer_idx=int(raw["layer_idx"]),
                head_idx=int(raw["head_idx"]),
                model_id=str(raw.get("model_id", data.get("model_id", "unknown"))),
            )
        edges.append(CausalEdge(
            head=head,
            behavior_category=str(raw["behavior_category"]),
            causal_effect=float(raw["causal_effect"]),
            ablation_delta=float(raw["ablation_delta"]),
            patching_delta=float(raw.get("patching_delta", 0.0)),
            confidence=float(raw.get("confidence", 1.0)),
            discovered_at=str(raw.get("discovered_at", "")),
        ))

    # Rebuild top_k_heads from loaded edges
    categories = list({e.behavior_category for e in edges})
    top_k: Dict[str, List[HeadSignature]] = {}
    for category in categories:
        cat_edges = sorted(
            [e for e in edges if e.behavior_category == category],
            key=lambda e: e.causal_effect,
            reverse=True,
        )
        top_k[category] = [e.head for e in cat_edges[:5]]

    return CausalGraph(
        model_id=str(data.get("model_id", "unknown")),
        edges=edges,
        top_k_heads=top_k,
        version=int(data.get("version", 1)),
    )


# ── Mock signal construction ──────────────────────────────────────────────────

def _build_safe_signal(idx: int) -> SimpleNamespace:
    """Construct a mock 'safe' SafetySignal-like object.

    Args:
        idx: Sequential index (used to vary values slightly).

    Returns:
        SimpleNamespace with composite_risk_score, probe_scores, and request_id.
    """
    base_score = float(_RNG.uniform(0.2, 0.4))
    probe_scores: Dict[str, float] = {
        "hallucination": float(_RNG.uniform(0.05, 0.2)),
        "jailbreak": float(_RNG.uniform(0.05, 0.2)),
        "toxic_reasoning": float(_RNG.uniform(0.05, 0.15)),
        "policy_violation": float(_RNG.uniform(0.05, 0.15)),
    }
    return SimpleNamespace(
        composite_risk_score=base_score,
        probe_scores=probe_scores,
        risk_level="safe",
        request_id=f"safe-{idx:04d}",
    )


def _build_unsafe_signal(idx: int) -> SimpleNamespace:
    """Construct a mock 'unsafe' SafetySignal-like object.

    Args:
        idx: Sequential index (used to vary values slightly).

    Returns:
        SimpleNamespace with composite_risk_score, probe_scores, and request_id.
    """
    base_score = float(_RNG.uniform(0.7, 0.95))
    probe_scores: Dict[str, float] = {
        "hallucination": float(_RNG.uniform(0.5, 0.9)),
        "jailbreak": float(_RNG.uniform(0.6, 0.95)),
        "toxic_reasoning": float(_RNG.uniform(0.4, 0.85)),
        "policy_violation": float(_RNG.uniform(0.3, 0.75)),
    }
    return SimpleNamespace(
        composite_risk_score=base_score,
        probe_scores=probe_scores,
        risk_level="critical",
        request_id=f"unsafe-{idx:04d}",
    )


# ── Verification logic ────────────────────────────────────────────────────────

def _run_correctness_checks(
    engine: CausalInterventionEngine,
    safe_signals: List[SimpleNamespace],
    unsafe_signals: List[SimpleNamespace],
    assert_delta_min: float,
) -> Tuple[int, int, int, int, List[Dict[str, Any]]]:
    """Verify engine behaviour on safe and unsafe signals.

    For each unsafe signal:  engine.maybe_intervene must return InterventionResult.
    For each safe signal:    engine.maybe_intervene must return None.

    Args:
        engine: Initialised CausalInterventionEngine.
        safe_signals: List of safe mock signals.
        unsafe_signals: List of unsafe mock signals.
        assert_delta_min: Minimum expected risk delta for unsafe interventions.

    Returns:
        Tuple of (unsafe_pass, unsafe_fail, safe_pass, safe_fail, rows) where
        rows is a list of result dicts for the verification table.
    """
    unsafe_pass = unsafe_fail = safe_pass = safe_fail = 0
    rows: List[Dict[str, Any]] = []

    for sig in unsafe_signals:
        try:
            result = engine.maybe_intervene(sig)
        except InterventionError as exc:
            logger.error(
                "intervention_error",
                request_id=sig.request_id,
                error=str(exc),
            )
            unsafe_fail += 1
            rows.append({
                "request_id": sig.request_id,
                "type": "UNSAFE",
                "risk_before": sig.composite_risk_score,
                "risk_after": None,
                "delta": None,
                "result_type": "ERROR",
                "pass": False,
            })
            continue

        if result is None:
            logger.warning(
                "unexpected_none_for_unsafe_signal",
                request_id=sig.request_id,
                score=sig.composite_risk_score,
            )
            unsafe_fail += 1
            rows.append({
                "request_id": sig.request_id,
                "type": "UNSAFE",
                "risk_before": sig.composite_risk_score,
                "risk_after": None,
                "delta": None,
                "result_type": "None (unexpected)",
                "pass": False,
            })
        elif not isinstance(result, InterventionResult):
            logger.warning(
                "wrong_return_type_for_unsafe_signal",
                request_id=sig.request_id,
                actual_type=type(result).__name__,
            )
            unsafe_fail += 1
            rows.append({
                "request_id": sig.request_id,
                "type": "UNSAFE",
                "risk_before": sig.composite_risk_score,
                "risk_after": None,
                "delta": None,
                "result_type": f"Wrong type: {type(result).__name__}",
                "pass": False,
            })
        else:
            delta_ok = result.delta >= assert_delta_min
            if delta_ok:
                unsafe_pass += 1
            else:
                unsafe_fail += 1
                logger.warning(
                    "delta_below_minimum",
                    request_id=sig.request_id,
                    delta=result.delta,
                    minimum=assert_delta_min,
                )
            rows.append({
                "request_id": sig.request_id,
                "type": "UNSAFE",
                "risk_before": result.spec.risk_score_before,
                "risk_after": result.risk_score_after,
                "delta": result.delta,
                "result_type": "InterventionResult",
                "pass": delta_ok,
            })

    for sig in safe_signals:
        try:
            result = engine.maybe_intervene(sig)
        except InterventionError as exc:
            logger.error(
                "intervention_error_on_safe_signal",
                request_id=sig.request_id,
                error=str(exc),
            )
            safe_fail += 1
            rows.append({
                "request_id": sig.request_id,
                "type": "SAFE",
                "risk_before": sig.composite_risk_score,
                "risk_after": None,
                "delta": None,
                "result_type": "ERROR",
                "pass": False,
            })
            continue

        if result is None:
            safe_pass += 1
            rows.append({
                "request_id": sig.request_id,
                "type": "SAFE",
                "risk_before": sig.composite_risk_score,
                "risk_after": None,
                "delta": None,
                "result_type": "None (correct)",
                "pass": True,
            })
        else:
            safe_fail += 1
            logger.warning(
                "unexpected_intervention_on_safe_signal",
                request_id=sig.request_id,
                score=sig.composite_risk_score,
            )
            rows.append({
                "request_id": sig.request_id,
                "type": "SAFE",
                "risk_before": sig.composite_risk_score,
                "risk_after": getattr(result, "risk_score_after", None),
                "delta": getattr(result, "delta", None),
                "result_type": f"Unexpected {type(result).__name__}",
                "pass": False,
            })

    return unsafe_pass, unsafe_fail, safe_pass, safe_fail, rows


def _run_latency_benchmark(
    engine: CausalInterventionEngine,
    n_iterations: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark prepare_intervention() CPU latency over many calls.

    Builds a representative unsafe signal for each call so the engine exercises
    real head-lookup logic rather than short-circuiting on an empty graph.

    Args:
        engine: Initialised CausalInterventionEngine.
        n_iterations: Number of calls to time.

    Returns:
        Tuple of (p50_ms, p95_ms, p99_ms).
    """
    latencies_ms: List[float] = []
    categories = CATEGORIES[:2]  # jailbreak + hallucination

    for i in range(n_iterations):
        risk_score = float(_RNG.uniform(0.7, 0.95))
        t0 = time.perf_counter()
        _ = engine.prepare_intervention(
            risk_score=risk_score,
            triggered_categories=categories,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        latencies_ms.append(elapsed_ms)

    arr = np.array(latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return p50, p95, p99


# ── Reporting helpers ─────────────────────────────────────────────────────────

def _print_verification_table(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted verification table to stdout.

    Args:
        rows: List of result dicts produced by _run_correctness_checks.
    """
    header = (
        f"{'Request ID':<18} {'Type':<8} {'Before':>7} {'After':>7} "
        f"{'Delta':>7} {'Result':<28} {'Pass':<5}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        before = f"{row['risk_before']:.3f}" if row["risk_before"] is not None else "  N/A"
        after = f"{row['risk_after']:.3f}" if row["risk_after"] is not None else "  N/A"
        delta = f"{row['delta']:.3f}" if row["delta"] is not None else "  N/A"
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{row['request_id']:<18} {row['type']:<8} {before:>7} {after:>7} "
            f"{delta:>7} {row['result_type']:<28} {status:<5}"
        )
    print(sep)


def _print_summary(
    unsafe_pass: int,
    unsafe_fail: int,
    safe_pass: int,
    safe_fail: int,
) -> None:
    """Print a correctness summary block to stdout.

    Args:
        unsafe_pass: Number of unsafe-signal tests that passed.
        unsafe_fail: Number of unsafe-signal tests that failed.
        safe_pass: Number of safe-signal tests that passed.
        safe_fail: Number of safe-signal tests that failed.
    """
    total = unsafe_pass + unsafe_fail + safe_pass + safe_fail
    total_pass = unsafe_pass + safe_pass
    print()
    print("=== Correctness Summary ===")
    print(f"  Unsafe signals  : {unsafe_pass + unsafe_fail} total | "
          f"{unsafe_pass} pass | {unsafe_fail} fail")
    print(f"  Safe signals    : {safe_pass + safe_fail} total | "
          f"{safe_pass} pass | {safe_fail} fail")
    print(f"  Overall         : {total_pass}/{total} passed")
    print()


def _print_sla_result(p50: float, p95: float, p99: float, budget_ms: float = 5.0) -> None:
    """Print the latency benchmark SLA result to stdout.

    Args:
        p50: 50th-percentile latency in milliseconds.
        p95: 95th-percentile latency in milliseconds.
        p99: 99th-percentile latency in milliseconds.
        budget_ms: SLA budget in milliseconds (default 5 ms).
    """
    sla_ok = p95 < budget_ms
    status = "PASS" if sla_ok else "FAIL"
    print("=== Latency Benchmark (prepare_intervention, 100 calls, CPU) ===")
    print(f"  p50 : {p50:.3f} ms")
    print(f"  p95 : {p95:.3f} ms  (budget: <{budget_ms} ms)  [{status}]")
    print(f"  p99 : {p99:.3f} ms")
    print()


# ── MLflow logging ────────────────────────────────────────────────────────────

def _log_to_mlflow(
    unsafe_pass: int,
    unsafe_fail: int,
    safe_pass: int,
    safe_fail: int,
    p50: float,
    p95: float,
    p99: float,
    graph_source: str,
    threshold: float,
) -> None:
    """Log verification results to MLflow.

    Falls back gracefully when MLflow is not configured.

    Args:
        unsafe_pass: Count of passing unsafe-signal tests.
        unsafe_fail: Count of failing unsafe-signal tests.
        safe_pass: Count of passing safe-signal tests.
        safe_fail: Count of failing safe-signal tests.
        p50: Latency p50 in milliseconds.
        p95: Latency p95 in milliseconds.
        p99: Latency p99 in milliseconds.
        graph_source: Description of how the causal graph was obtained.
        threshold: Intervention threshold used in this run.
    """
    try:
        import mlflow  # noqa: PLC0415

        total = unsafe_pass + unsafe_fail + safe_pass + safe_fail
        total_pass = unsafe_pass + safe_pass

        with mlflow.start_run(run_name="intervention_verification"):
            mlflow.set_tags({
                "module": "causal_engine",
                "verification_type": "intervention_correctness",
                "graph_source": graph_source,
            })
            mlflow.log_params({
                "intervention_threshold": threshold,
                "n_unsafe_signals": unsafe_pass + unsafe_fail,
                "n_safe_signals": safe_pass + safe_fail,
            })
            mlflow.log_metrics({
                "unsafe_pass": float(unsafe_pass),
                "unsafe_fail": float(unsafe_fail),
                "safe_pass": float(safe_pass),
                "safe_fail": float(safe_fail),
                "total_pass_rate": total_pass / max(total, 1),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "sla_ok": float(p95 < 5.0),
            })
        logger.info("mlflow_run_logged", run_name="intervention_verification")

    except ImportError:
        logger.warning(
            "mlflow_unavailable",
            hint="pip install mlflow — skipping MLflow logging",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "mlflow_logging_failed",
            error=str(exc),
            hint="Check MLFLOW_TRACKING_URI environment variable",
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Verify CausalInterventionEngine correctness and latency SLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--causal-graph-path",
        type=Path,
        default=Path("models/causal/llama31_8b_causal_graph.json"),
        help="Path to a serialised CausalGraph JSON file. "
             "If absent, a synthetic graph is built automatically.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Total number of mock signals to generate (half safe, half unsafe).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Intervention threshold for CausalInterventionEngine.",
    )
    parser.add_argument(
        "--assert-delta-min",
        type=float,
        default=0.2,
        help="Minimum required risk-score delta for a successful intervention.",
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=100,
        help="Number of prepare_intervention() calls for the latency benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the intervention verification harness.

    Returns:
        Exit code: 0 on full pass, 1 on any correctness failure or SLA breach.
    """
    args = _parse_args()

    logger.info(
        "verify_interventions_start",
        causal_graph_path=str(args.causal_graph_path),
        n_samples=args.n_samples,
        threshold=args.threshold,
        assert_delta_min=args.assert_delta_min,
    )

    # ── Build or load causal graph ─────────────────────────────────────────────
    graph_path: Path = args.causal_graph_path
    if graph_path.exists():
        graph = _load_causal_graph_from_json(graph_path)
        graph_source = str(graph_path)
        print(f"\nLoaded CausalGraph from: {graph_path}")
    else:
        logger.warning(
            "causal_graph_not_found",
            path=str(graph_path),
            action="building_synthetic_graph",
        )
        print(f"\nCausal graph not found at '{graph_path}' — using synthetic graph.")
        graph = _make_synthetic_causal_graph()
        graph_source = "synthetic"

    print(
        f"CausalGraph: model_id={graph.model_id!r}, "
        f"edges={len(graph.edges)}, version={graph.version}\n"
    )

    # ── Initialise engine ──────────────────────────────────────────────────────
    engine = CausalInterventionEngine(
        causal_graph=graph,
        intervention_threshold=args.threshold,
        default_scale_factor=0.25,
    )

    # ── Build mock signals ─────────────────────────────────────────────────────
    n_each = args.n_samples // 2
    safe_signals = [_build_safe_signal(i) for i in range(n_each)]
    unsafe_signals = [_build_unsafe_signal(i) for i in range(n_each)]

    logger.info(
        "mock_signals_created",
        n_safe=len(safe_signals),
        n_unsafe=len(unsafe_signals),
    )

    # ── Correctness verification ───────────────────────────────────────────────
    print("=== Intervention Verification Table ===\n")
    (
        unsafe_pass,
        unsafe_fail,
        safe_pass,
        safe_fail,
        rows,
    ) = _run_correctness_checks(
        engine=engine,
        safe_signals=safe_signals,
        unsafe_signals=unsafe_signals,
        assert_delta_min=args.assert_delta_min,
    )
    _print_verification_table(rows)
    _print_summary(unsafe_pass, unsafe_fail, safe_pass, safe_fail)

    # ── Latency benchmark ──────────────────────────────────────────────────────
    print("Running latency benchmark …")
    p50, p95, p99 = _run_latency_benchmark(engine, n_iterations=args.latency_iterations)
    _print_sla_result(p50, p95, p99, budget_ms=5.0)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    _log_to_mlflow(
        unsafe_pass=unsafe_pass,
        unsafe_fail=unsafe_fail,
        safe_pass=safe_pass,
        safe_fail=safe_fail,
        p50=p50,
        p95=p95,
        p99=p99,
        graph_source=graph_source,
        threshold=args.threshold,
    )

    # ── Final assertions ───────────────────────────────────────────────────────
    all_correct = (unsafe_fail == 0) and (safe_fail == 0)
    sla_ok = p95 < 5.0

    if not all_correct:
        logger.error(
            "correctness_verification_failed",
            unsafe_fail=unsafe_fail,
            safe_fail=safe_fail,
        )
        print("RESULT: FAIL — correctness checks did not pass.")

    if not sla_ok:
        logger.error(
            "sla_violation",
            p95_ms=p95,
            budget_ms=5.0,
        )
        print(f"RESULT: FAIL — p95 latency {p95:.3f}ms exceeds 5ms SLA.")

    if all_correct and sla_ok:
        print("RESULT: PASS — all correctness checks passed and SLA met.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
