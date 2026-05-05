"""
MOD-01: LatentSentinel Latency Benchmark
==========================================
Measures hook-to-signal latency for LatentSentinel under production-like load.
Validates the <10ms p95 SLA on an 8B model.

Run with:
    python scripts/benchmark_sentinel.py
    python scripts/benchmark_sentinel.py --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --num-requests 1000 --device cuda

Output:
    - Console table with p50/p95/p99 latency
    - docs/benchmarks/sentinel_latency.md
    - MLflow run with all metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_sentinel_config
from src.latent_sentinel.sentinel import (
    LatentSentinel,
    LinearResidualProbe,
    ProbeCategory,
    RiskLevel,
)

logger = structlog.get_logger(__name__)


# ── Stub model for CPU testing without a real LLM ────────────────────────────

class _BenchmarkInner(nn.Module):
    """Inner model with .layers attribute — matches HuggingFace Llama layout expected
    by HookManager._get_target_layers() which checks for model.model.layers."""

    def __init__(self, num_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BenchmarkTransformer(nn.Module):
    """A minimal transformer that mimics Llama 3.1 8B output shapes.

    Used when --stub-model is passed (CPU benchmarking, no GPU required).
    Hidden dim 4096 matches Llama 3.1 8B.

    The nested ``self.model.layers`` structure mirrors HuggingFace LLaMA so that
    ``HookManager._get_target_layers()`` can discover the layers automatically.

    Args:
        num_layers: Number of transformer layers to simulate.
        hidden_dim: Hidden dimension (4096 for 8B models).
        seq_len: Sequence length per forward pass.
    """

    def __init__(
        self,
        num_layers: int = 32,
        hidden_dim: int = 4096,
        seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        # Nested so HookManager._get_target_layers() finds model.model.layers
        self.model = _BenchmarkInner(num_layers, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through all layers, emitting hooks on each.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Output tensor of same shape.
        """
        return self.model(x)


def build_stub_probes(
    target_layers: List[int],
    probe_dim: int = 128,
    input_dim: int = 4096,
) -> Dict[ProbeCategory, LinearResidualProbe]:
    """Build untrained stub probes for benchmarking (no weights needed).

    Args:
        target_layers: Unused — kept for API compatibility.
        probe_dim: Internal probe projection dimension (default 128).
        input_dim: Hidden dimension of the model activations (e.g. 4096 for Llama 8B).
                   Must match the model's residual stream width.

    Returns:
        Flat dict {ProbeCategory → probe} used by LatentSentinel.
    """
    probes: Dict[ProbeCategory, LinearResidualProbe] = {}
    for category in ProbeCategory:
        probe = LinearResidualProbe(
            hidden_dim=input_dim,   # activation width, NOT the probe's internal dim
            category=category,
            probe_dim=probe_dim,
        )
        probe.eval()
        probes[category] = probe
    return probes


# ── Core benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(
    model: nn.Module,
    sentinel: LatentSentinel,
    num_requests: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    device: str,
    warmup_requests: int = 50,
) -> Dict[str, float]:
    """Run the latency benchmark.

    Args:
        model: Model with LatentSentinel hooks attached.
        sentinel: The LatentSentinel instance being benchmarked.
        num_requests: Total number of forward passes to time.
        batch_size: Batch size per forward pass.
        seq_len: Sequence length per batch.
        hidden_dim: Hidden dimension of the model.
        device: Torch device string.
        warmup_requests: Number of warm-up forward passes (not timed).

    Returns:
        Dict with latency percentiles and throughput metrics.
    """
    model.eval()
    latencies_ms: List[float] = []

    dummy_input = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warm up (fill caches, JIT compile CUDA kernels)
    logger.info("benchmark_warmup", warmup_requests=warmup_requests)
    with torch.no_grad():
        for _ in range(warmup_requests):
            _ = model(dummy_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed loop
    logger.info("benchmark_start", num_requests=num_requests)
    with torch.no_grad():
        for i in range(num_requests):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

            if (i + 1) % 100 == 0:
                so_far = np.array(latencies_ms)
                logger.info(
                    "benchmark_progress",
                    completed=i + 1,
                    p95_ms=f"{np.percentile(so_far, 95):.2f}",
                )

    arr = np.array(latencies_ms)
    results = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
        "min_ms": float(arr.min()),
        "std_ms": float(arr.std()),
        "num_requests": num_requests,
        "sla_pass": float(np.percentile(arr, 95)) < 10.0,
    }
    return results


# ── Report generation ─────────────────────────────────────────────────────────

def write_report(results: Dict[str, float], model_name: str, device: str) -> Path:
    """Write a markdown benchmark report to docs/benchmarks/.

    Args:
        results: Latency metrics dict from run_benchmark().
        model_name: Model identifier string.
        device: Device used for benchmarking.

    Returns:
        Path to the generated markdown file.
    """
    report_dir = Path("docs/benchmarks")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sentinel_latency.md"

    sla_status = "PASS" if results["sla_pass"] else "FAIL"

    content = f"""# LatentSentinel Latency Benchmark

**Model:** `{model_name}`
**Device:** `{device}`
**Requests:** {results['num_requests']:,}
**SLA Target:** <10ms p95
**SLA Status:** {sla_status}

## Results

| Metric | Value |
|--------|-------|
| p50 | {results['p50_ms']:.2f} ms |
| p90 | {results['p90_ms']:.2f} ms |
| **p95** | **{results['p95_ms']:.2f} ms** |
| p99 | {results['p99_ms']:.2f} ms |
| Mean | {results['mean_ms']:.2f} ms |
| Std Dev | {results['std_ms']:.2f} ms |
| Min | {results['min_ms']:.2f} ms |
| Max | {results['max_ms']:.2f} ms |

## SLA Assessment

The p95 latency of **{results['p95_ms']:.2f}ms** is {'within' if results['sla_pass'] else 'above'} the 10ms SLA budget.
ARGUS overhead {'does not impact' if results['sla_pass'] else 'exceeds the budget for'} production LLM throughput.
"""
    report_path.write_text(content, encoding="utf-8")
    logger.info("benchmark_report_written", path=str(report_path))
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LatentSentinel hook latency")
    parser.add_argument(
        "--model",
        default="stub",
        help="HuggingFace model name, local path, or 'stub' for CPU stub model",
    )
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probe-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--mlflow-experiment", default="argus/benchmarks")
    parser.add_argument(
        "--assert-sla",
        action="store_true",
        help="Exit with code 1 if p95 > 10ms (for CI gates)",
    )
    args = parser.parse_args()

    cfg = load_sentinel_config()

    # Determine hidden_dim
    hidden_dim = 4096  # Llama 3.1 8B
    if "gemma" in args.model.lower():
        hidden_dim = 3072

    # Build or load model
    if args.model == "stub":
        logger.info("using_stub_model", hidden_dim=hidden_dim)
        model = BenchmarkTransformer(
            num_layers=32,
            hidden_dim=hidden_dim,
            seq_len=args.seq_len,
        ).to(args.device)
    else:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("Run: pip install transformers")
        logger.info("loading_model", model=args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )
        model.eval()

    # Build stub probes
    stub_probes = build_stub_probes(
        target_layers=cfg.hooks.target_layers,
        probe_dim=args.probe_dim,
        input_dim=hidden_dim,
    )

    # Build LatentSentinel and attach hooks
    sentinel = LatentSentinel(
        probes=stub_probes,
        target_layers=cfg.hooks.target_layers,
        device=args.device,
    )
    sentinel.monitor(model)

    # Run benchmark
    results = run_benchmark(
        model=model,
        sentinel=sentinel,
        num_requests=args.num_requests,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=hidden_dim,
        device=args.device,
        warmup_requests=args.warmup,
    )

    # MLflow logging
    try:
        import mlflow
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="sentinel_benchmark"):
            mlflow.log_params(
                {"model": args.model, "device": args.device, "num_requests": args.num_requests}
            )
            mlflow.log_metrics(results)
    except Exception as e:
        logger.warning("mlflow_logging_failed", error=str(e))

    # Write report
    report_path = write_report(results, model_name=args.model, device=args.device)

    # Print summary
    sla_status = "[PASS]" if results["sla_pass"] else "[FAIL]"
    print(f"\n=== LatentSentinel Benchmark Results ===")
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device}")
    print(f"Requests:    {args.num_requests:,}")
    print(f"")
    print(f"p50 latency: {results['p50_ms']:.2f} ms")
    print(f"p95 latency: {results['p95_ms']:.2f} ms  (SLA target: <10ms)  {sla_status}")
    print(f"p99 latency: {results['p99_ms']:.2f} ms")
    print(f"")
    print(f"Report:      {report_path}")

    if args.assert_sla and not results["sla_pass"]:
        logger.error(
            "sla_violation",
            p95_ms=results["p95_ms"],
            budget_ms=10.0,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
