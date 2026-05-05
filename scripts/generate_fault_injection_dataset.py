"""
MOD-05: Fault Injection Dataset Generator
==========================================
Generates synthetic telemetry windows labeled with failure events for
training the PredictiveOracle. Simulates 1000+ failure episodes covering
the 4 main failure modes observed in production LLM deployments.

9 telemetry features per timestep (matching PredictiveOracleConfig):
    0: request_rate         — requests per second
    1: error_rate           — fraction of requests with errors
    2: p95_latency_ms       — 95th percentile response latency
    3: token_throughput     — tokens generated per second
    4: safety_score_avg     — rolling average safety score [0,1]
    5: memory_utilization   — GPU/CPU memory fraction [0,1]
    6: queue_depth          — pending request queue depth
    7: cpu_utilization      — CPU utilization fraction [0,1]
    8: context_window_fill  — fraction of context window used [0,1]

Output:
    data/oracle/fault_injection_train.npz
    data/oracle/fault_injection_val.npz
    data/oracle/fault_injection_test.npz

Run with:
    python scripts/generate_fault_injection_dataset.py
    python scripts/generate_fault_injection_dataset.py --n-episodes 2000 --seq-len 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger(__name__)

# ── Feature indices ───────────────────────────────────────────────────────────

N_FEATURES = 9
IDX_REQUEST_RATE = 0
IDX_ERROR_RATE = 1
IDX_P95_LATENCY = 2
IDX_TOKEN_THROUGHPUT = 3
IDX_SAFETY_SCORE = 4
IDX_MEMORY_UTIL = 5
IDX_QUEUE_DEPTH = 6
IDX_CPU_UTIL = 7
IDX_CONTEXT_FILL = 8

# ── Normal operating baseline ─────────────────────────────────────────────────

BASELINE_MEAN = np.array([
    50.0,   # request_rate (req/s)
    0.02,   # error_rate (2%)
    120.0,  # p95_latency_ms
    2000.0, # token_throughput
    0.15,   # safety_score_avg (mostly safe)
    0.60,   # memory_utilization
    5.0,    # queue_depth
    0.45,   # cpu_utilization
    0.40,   # context_window_fill
], dtype=np.float32)

BASELINE_STD = np.array([
    10.0,   # request_rate noise
    0.005,  # error_rate noise
    15.0,   # latency noise
    200.0,  # throughput noise
    0.02,   # safety noise
    0.05,   # memory noise
    2.0,    # queue noise
    0.05,   # cpu noise
    0.05,   # context noise
], dtype=np.float32)


def _generate_normal_window(seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a normal (no-failure) telemetry window.

    Args:
        seq_len: Number of timesteps.
        rng: NumPy random generator.

    Returns:
        Array of shape (seq_len, N_FEATURES).
    """
    window = rng.normal(
        loc=BASELINE_MEAN,
        scale=BASELINE_STD,
        size=(seq_len, N_FEATURES),
    ).astype(np.float32)
    # Clamp to physically valid ranges
    window[:, IDX_ERROR_RATE] = np.clip(window[:, IDX_ERROR_RATE], 0, 1)
    window[:, IDX_SAFETY_SCORE] = np.clip(window[:, IDX_SAFETY_SCORE], 0, 1)
    window[:, IDX_MEMORY_UTIL] = np.clip(window[:, IDX_MEMORY_UTIL], 0, 1)
    window[:, IDX_CPU_UTIL] = np.clip(window[:, IDX_CPU_UTIL], 0, 1)
    window[:, IDX_CONTEXT_FILL] = np.clip(window[:, IDX_CONTEXT_FILL], 0, 1)
    window[:, IDX_REQUEST_RATE] = np.clip(window[:, IDX_REQUEST_RATE], 0, None)
    window[:, IDX_QUEUE_DEPTH] = np.clip(window[:, IDX_QUEUE_DEPTH], 0, None)
    window[:, IDX_P95_LATENCY] = np.clip(window[:, IDX_P95_LATENCY], 0, None)
    window[:, IDX_TOKEN_THROUGHPUT] = np.clip(window[:, IDX_TOKEN_THROUGHPUT], 0, None)
    return window


def _inject_prompt_injection_attack(
    window: np.ndarray, onset_frac: float = 0.6
) -> np.ndarray:
    """Inject prompt injection attack signature.

    Signature: safety score spikes + error rate jumps + latency increases.

    Args:
        window: Baseline window (seq_len, N_FEATURES).
        onset_frac: Attack starts at this fraction of the window.

    Returns:
        Modified window with attack signature.
    """
    seq_len = window.shape[0]
    onset = int(seq_len * onset_frac)
    attack = window.copy()

    ramp = np.linspace(0, 1, seq_len - onset)
    attack[onset:, IDX_SAFETY_SCORE] = np.clip(
        attack[onset:, IDX_SAFETY_SCORE] + ramp * 0.7, 0, 1
    )
    attack[onset:, IDX_ERROR_RATE] = np.clip(
        attack[onset:, IDX_ERROR_RATE] + ramp * 0.15, 0, 1
    )
    attack[onset:, IDX_P95_LATENCY] *= (1 + ramp * 2.0)
    return attack


def _inject_context_overflow(
    window: np.ndarray, onset_frac: float = 0.5
) -> np.ndarray:
    """Inject context window overflow failure signature.

    Signature: context fill grows toward 1.0 → latency explodes → OOM errors.

    Args:
        window: Baseline window.
        onset_frac: Overflow starts at this fraction.

    Returns:
        Modified window.
    """
    seq_len = window.shape[0]
    onset = int(seq_len * onset_frac)
    overflow = window.copy()

    ramp = np.linspace(0, 1, seq_len - onset) ** 1.5
    overflow[onset:, IDX_CONTEXT_FILL] = np.clip(
        overflow[onset:, IDX_CONTEXT_FILL] + ramp * 0.55, 0, 1
    )
    overflow[onset:, IDX_P95_LATENCY] *= (1 + ramp * 5.0)
    overflow[onset:, IDX_MEMORY_UTIL] = np.clip(
        overflow[onset:, IDX_MEMORY_UTIL] + ramp * 0.35, 0, 1
    )
    overflow[onset:, IDX_ERROR_RATE] = np.clip(
        overflow[onset:, IDX_ERROR_RATE] + ramp * 0.25, 0, 1
    )
    return overflow


def _inject_rate_spike(
    window: np.ndarray, onset_frac: float = 0.55
) -> np.ndarray:
    """Inject traffic rate spike leading to queue saturation.

    Signature: request rate spikes → queue depth grows → latency degrades.

    Args:
        window: Baseline window.
        onset_frac: Spike starts at this fraction.

    Returns:
        Modified window.
    """
    seq_len = window.shape[0]
    onset = int(seq_len * onset_frac)
    spike = window.copy()

    ramp = np.linspace(0, 1, seq_len - onset)
    spike[onset:, IDX_REQUEST_RATE] *= (1 + ramp * 8.0)
    spike[onset:, IDX_QUEUE_DEPTH] += ramp * 80.0
    spike[onset:, IDX_P95_LATENCY] *= (1 + ramp * 3.5)
    spike[onset:, IDX_CPU_UTIL] = np.clip(
        spike[onset:, IDX_CPU_UTIL] + ramp * 0.45, 0, 1
    )
    spike[onset:, IDX_TOKEN_THROUGHPUT] *= np.clip(1 - ramp * 0.6, 0.1, 1.0)
    return spike


def _inject_adversarial_input(
    window: np.ndarray, onset_frac: float = 0.65
) -> np.ndarray:
    """Inject adversarial input bombardment signature.

    Signature: safety scores climb steadily, error rate increases.

    Args:
        window: Baseline window.
        onset_frac: Attack starts at this fraction.

    Returns:
        Modified window.
    """
    seq_len = window.shape[0]
    onset = int(seq_len * onset_frac)
    adv = window.copy()

    ramp = np.linspace(0, 1, seq_len - onset)
    adv[onset:, IDX_SAFETY_SCORE] = np.clip(
        adv[onset:, IDX_SAFETY_SCORE] + ramp * 0.60, 0, 1
    )
    adv[onset:, IDX_ERROR_RATE] = np.clip(
        adv[onset:, IDX_ERROR_RATE] + ramp * 0.10, 0, 1
    )
    adv[onset:, IDX_TOKEN_THROUGHPUT] *= np.clip(1 - ramp * 0.4, 0.3, 1.0)
    return adv


FAULT_INJECTORS = [
    _inject_prompt_injection_attack,
    _inject_context_overflow,
    _inject_rate_spike,
    _inject_adversarial_input,
]


def generate_dataset(
    n_episodes: int,
    seq_len: int,
    fault_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of telemetry windows with failure labels.

    Args:
        n_episodes: Total number of episodes.
        seq_len: Timesteps per episode.
        fault_fraction: Fraction of episodes that contain a failure.
        rng: NumPy random generator.

    Returns:
        Tuple of (windows, labels):
            windows: shape (n_episodes, seq_len, N_FEATURES)
            labels: shape (n_episodes,) — 1 = failure in the next horizon
    """
    n_fault = int(n_episodes * fault_fraction)
    n_normal = n_episodes - n_fault

    windows = []
    labels = []

    # Normal episodes
    for _ in range(n_normal):
        w = _generate_normal_window(seq_len, rng)
        windows.append(w)
        labels.append(0)

    # Fault episodes
    for _ in range(n_fault):
        w = _generate_normal_window(seq_len, rng)
        injector = FAULT_INJECTORS[rng.integers(0, len(FAULT_INJECTORS))]
        onset_frac = rng.uniform(0.4, 0.8)
        w = injector(w, onset_frac=onset_frac)
        windows.append(w)
        labels.append(1)

    # Shuffle
    indices = np.arange(len(windows))
    rng.shuffle(indices)
    windows_arr = np.stack(windows)[indices]
    labels_arr = np.array(labels, dtype=np.int32)[indices]

    return windows_arr, labels_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fault injection dataset for PredictiveOracle")
    parser.add_argument("--n-episodes", type=int, default=1500)
    parser.add_argument("--seq-len", type=int, default=60, help="Timesteps per window (matches config)")
    parser.add_argument("--fault-fraction", type=float, default=0.40)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--output-dir", default="data/oracle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "generating_fault_injection_dataset",
        n_episodes=args.n_episodes,
        seq_len=args.seq_len,
        fault_fraction=args.fault_fraction,
    )

    windows, labels = generate_dataset(
        n_episodes=args.n_episodes,
        seq_len=args.seq_len,
        fault_fraction=args.fault_fraction,
        rng=rng,
    )

    # Split
    n = len(windows)
    n_test = int(n * args.test_split)
    n_val = int(n * args.val_split)
    n_train = n - n_val - n_test

    splits = {
        "train": (windows[:n_train], labels[:n_train]),
        "val": (windows[n_train:n_train + n_val], labels[n_train:n_train + n_val]),
        "test": (windows[n_train + n_val:], labels[n_train + n_val:]),
    }

    for split_name, (X, y) in splits.items():
        path = output_dir / f"fault_injection_{split_name}.npz"
        np.savez(path, windows=X, labels=y)
        fault_count = int(y.sum())
        logger.info(
            "split_saved",
            split=split_name,
            n=len(X),
            faults=fault_count,
            fault_pct=f"{100 * fault_count / max(len(X), 1):.1f}%",
            path=str(path),
        )

    # Print summary
    print(f"\n=== Fault Injection Dataset Generated ===")
    print(f"Total episodes : {n:,}")
    print(f"  Train        : {n_train:,} ({int(labels[:n_train].sum())} faults)")
    print(f"  Val          : {n_val:,} ({int(labels[n_train:n_train+n_val].sum())} faults)")
    print(f"  Test         : {n_test:,} ({int(labels[n_train+n_val:].sum())} faults)")
    print(f"Window shape   : ({args.seq_len}, {N_FEATURES})")
    print(f"Fault types    : {len(FAULT_INJECTORS)} (injection, overflow, rate spike, adversarial)")
    print(f"Output         : {output_dir}")


if __name__ == "__main__":
    main()
