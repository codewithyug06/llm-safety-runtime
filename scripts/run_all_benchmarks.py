"""
ARGUS Full Benchmark Suite
===========================
Orchestrates all benchmark and evaluation targets in sequence and writes a
consolidated Markdown report to docs/benchmarks/full_report.md.

Run with:
    python scripts/run_all_benchmarks.py
    python scripts/run_all_benchmarks.py --output docs/benchmarks/full_report.md
    python scripts/run_all_benchmarks.py --skip-slow  # Skip GPU-dependent steps

Triggered via: make benchmark (full suite)
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent


# ── Benchmark step definitions ────────────────────────────────────────────────

@dataclass
class BenchmarkStep:
    """A single benchmark or evaluation step."""

    name: str
    description: str
    command: List[str]
    sla_check: str  # Human-readable SLA description
    requires_gpu: bool = False
    requires_data: bool = True
    timeout_seconds: int = 300


BENCHMARK_STEPS: List[BenchmarkStep] = [
    # ── MOD-01: LatentSentinel ────────────────────────────────────────────
    BenchmarkStep(
        name="sentinel_latency",
        description="MOD-01 LatentSentinel - p95 hook latency (<10ms on GPU, reported on CPU)",
        command=[sys.executable, "scripts/benchmark_sentinel.py",
                 "--num-requests", "500", "--warmup", "20"],
        sla_check="p95 latency < 10ms (GPU) / informational on CPU",
        requires_gpu=False,
        requires_data=False,
        timeout_seconds=120,
    ),
    BenchmarkStep(
        name="probe_accuracy",
        description="MOD-01 LatentSentinel - probe precision/recall (>85% precision target)",
        command=[sys.executable, "scripts/eval_probes.py",
                 "--target-precision", "0.85", "--target-f1", "0.82"],
        sla_check="precision >= 85%, F1 >= 82%",
        requires_gpu=False,
        requires_data=True,
        timeout_seconds=300,
    ),
    # ── MOD-03: OmniSafetyCritic ─────────────────────────────────────────
    BenchmarkStep(
        name="critic_accuracy",
        description="MOD-03 OmniSafetyCritic - precision/latency (<80ms, >85% precision)",
        command=[sys.executable, "scripts/eval_critic.py",
                 "--target-precision", "0.85"],
        sla_check="precision >= 85%, p95 < 80ms",
        requires_gpu=True,
        requires_data=True,
        timeout_seconds=600,
    ),
    # ── MOD-04: FederatedRLHF ─────────────────────────────────────────────
    BenchmarkStep(
        name="federated_gap",
        description="MOD-04 FederatedRLHF - accuracy gap vs centralized (<5%)",
        command=[sys.executable, "scripts/eval_federated.py",
                 "--target-gap", "0.05", "--target-epsilon", "3.0"],
        sla_check="accuracy gap < 5%, eps < 3.0",
        requires_gpu=False,
        requires_data=True,
        timeout_seconds=300,
    ),
    # ── MOD-05: PredictiveOracle ─────────────────────────────────────────
    BenchmarkStep(
        name="oracle_accuracy",
        description="MOD-05 PredictiveOracle - F1 at 60s horizon (>78% target)",
        command=[sys.executable, "scripts/eval_oracle.py",
                 "--target-f1", "0.78"],
        sla_check="F1 >= 78% @ 60s horizon",
        requires_gpu=False,
        requires_data=True,
        timeout_seconds=300,
    ),
]


# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Result of a single benchmark step execution."""

    step: BenchmarkStep
    status: str  # "passed" | "failed" | "skipped" | "error"
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    skip_reason: str = ""


# ── Execution engine ──────────────────────────────────────────────────────────

def run_step(step: BenchmarkStep, skip_gpu: bool = False) -> StepResult:
    """Execute a benchmark step and capture the result.

    Args:
        step: The benchmark step to execute.
        skip_gpu: If True, skip GPU-dependent steps gracefully.

    Returns:
        StepResult with status and captured output.
    """
    if step.requires_gpu and skip_gpu:
        logger.info("skipping_gpu_step", name=step.name)
        return StepResult(
            step=step,
            status="skipped",
            skip_reason="GPU not available (--skip-slow mode)",
        )

    logger.info("running_benchmark", name=step.name, command=" ".join(step.command))
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            step.command,
            capture_output=True,
            text=True,
            timeout=step.timeout_seconds,
            cwd=str(BASE_DIR),
        )
        duration = time.perf_counter() - t0

        if result.returncode == 0:
            status = "passed"
        else:
            status = "failed"

        return StepResult(
            step=step,
            status=status,
            duration_seconds=duration,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - t0
        logger.error("benchmark_timeout", name=step.name, timeout=step.timeout_seconds)
        return StepResult(
            step=step,
            status="error",
            duration_seconds=duration,
            stderr=f"Timed out after {step.timeout_seconds}s",
            return_code=-1,
        )
    except Exception as exc:
        duration = time.perf_counter() - t0
        logger.error("benchmark_error", name=step.name, error=str(exc))
        return StepResult(
            step=step,
            status="error",
            duration_seconds=duration,
            stderr=str(exc),
            return_code=-1,
        )


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    results: List[StepResult],
    output_path: Path,
    total_duration: float,
) -> None:
    """Write a Markdown benchmark report.

    Args:
        results: List of step results in execution order.
        output_path: Path to write the Markdown report.
        total_duration: Total wall-clock time for all steps.
    """
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    passed = sum(1 for r in results if r.status == "passed")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status in ("skipped", "error"))

    lines: List[str] = [
        "# ARGUS Full Benchmark Report",
        "",
        f"> Generated: {now}",
        f"> Total wall-clock time: {total_duration:.1f}s",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✅ Passed | {passed} |",
        f"| ❌ Failed | {failed} |",
        f"| ⏭ Skipped | {skipped} |",
        f"| **Total** | {len(results)} |",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Step | Description | SLA | Status | Duration |",
        "|------|-------------|-----|--------|----------|",
    ]

    for r in results:
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏭", "error": "⚠️"}.get(r.status, "?")
        dur = f"{r.duration_seconds:.1f}s" if r.duration_seconds > 0 else "—"
        lines.append(
            f"| `{r.step.name}` | {r.step.description} | {r.step.sla_check} "
            f"| {icon} {r.status.upper()} | {dur} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Reference Targets",
        "",
        "| Module | Metric | Target | Source |",
        "|--------|--------|--------|--------|",
        "| MOD-01 LatentSentinel | p95 hook latency | < 10ms | `benchmark_sentinel.py` |",
        "| MOD-01 LatentSentinel | Hallucination probe precision | > 85% | `eval_probes.py` |",
        "| MOD-03 OmniSafetyCritic | Precision | > 85% | `eval_critic.py` |",
        "| MOD-03 OmniSafetyCritic | p95 scoring latency | < 80ms | `eval_critic.py` |",
        "| MOD-04 FederatedRLHF | Accuracy gap vs centralized | < 5% | `eval_federated.py` |",
        "| MOD-04 FederatedRLHF | Privacy budget ε | < 3.0 | `eval_federated.py` |",
        "| MOD-05 PredictiveOracle | F1 at 60s horizon | > 78% | `eval_oracle.py` |",
        "| MOD-06 AutonomousRemediator | E2E remediation p99 | < 200ms | `tests/e2e/` |",
        "",
        "---",
        "",
        "## Step Details",
        "",
    ]

    for r in results:
        icon = {"passed": "✅", "failed": "❌", "skipped": "⏭", "error": "⚠️"}.get(r.status, "?")
        lines += [
            f"### {icon} {r.step.name}",
            "",
            f"**{r.step.description}**",
            "",
            f"- SLA: `{r.step.sla_check}`",
            f"- Status: **{r.status.upper()}**",
            f"- Duration: {r.duration_seconds:.2f}s",
            f"- Command: `{' '.join(r.step.command)}`",
        ]

        if r.skip_reason:
            lines += [f"- Skip reason: {r.skip_reason}"]

        if r.stdout:
            # Extract last 50 lines of stdout for conciseness
            stdout_lines = r.stdout.strip().splitlines()
            tail = stdout_lines[-50:] if len(stdout_lines) > 50 else stdout_lines
            lines += [
                "",
                "<details><summary>stdout</summary>",
                "",
                "```",
            ] + tail + [
                "```",
                "",
                "</details>",
            ]

        if r.stderr and r.status in ("failed", "error"):
            stderr_lines = r.stderr.strip().splitlines()
            tail = stderr_lines[-20:] if len(stderr_lines) > 20 else stderr_lines
            lines += [
                "",
                "<details><summary>stderr</summary>",
                "",
                "```",
            ] + tail + [
                "```",
                "",
                "</details>",
            ]

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report_written", path=str(output_path))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all ARGUS benchmarks")
    parser.add_argument(
        "--output",
        default="docs/benchmarks/full_report.md",
        help="Path to write the Markdown report",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip GPU-dependent benchmarks (for CI on CPU runners)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help="Run only specific steps by name (e.g. --steps sentinel_latency oracle_accuracy)",
    )
    args = parser.parse_args()

    # Filter steps
    steps_to_run = BENCHMARK_STEPS
    if args.steps:
        steps_to_run = [s for s in BENCHMARK_STEPS if s.name in args.steps]
        if not steps_to_run:
            print(f"No matching steps for: {args.steps}")
            print(f"Available: {[s.name for s in BENCHMARK_STEPS]}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  ARGUS Full Benchmark Suite")
    print(f"  Steps: {len(steps_to_run)}  |  Skip GPU: {args.skip_slow}")
    print(f"{'='*60}\n")

    results: List[StepResult] = []
    suite_start = time.perf_counter()

    for i, step in enumerate(steps_to_run, 1):
        print(f"[{i}/{len(steps_to_run)}] {step.name}")
        print(f"  {step.description}")

        result = run_step(step, skip_gpu=args.skip_slow)
        results.append(result)

        icon = {"passed": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]", "error": "[ERR]"}.get(
            result.status, "[?]"
        )
        print(f"  {icon} {result.status.upper()}  ({result.duration_seconds:.1f}s)\n")

        # Print last few lines of stdout inline
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-8:]:
                print(f"    {line}")
            print()

    total_duration = time.perf_counter() - suite_start

    # Generate report
    output_path = BASE_DIR / args.output
    generate_report(results, output_path, total_duration)

    # Final summary
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    errors = [r for r in results if r.status == "error"]
    skipped = [r for r in results if r.status == "skipped"]

    print(f"\n{'='*60}")
    print(f"  Benchmark Suite Complete  ({total_duration:.1f}s total)")
    print(f"{'='*60}")
    print(f"  [OK]   Passed : {len(passed)}")
    print(f"  [FAIL] Failed : {len(failed)}")
    print(f"  [ERR]  Errors : {len(errors)}")
    print(f"  [SKIP] Skipped: {len(skipped)}")
    print(f"\n  Report: {output_path}\n")

    if failed or errors:
        print("Failed steps:")
        for r in failed + errors:
            print(f"  - {r.step.name}: {r.status}")
        sys.exit(1)

    print("[OK] All benchmarks passed")


if __name__ == "__main__":
    main()
