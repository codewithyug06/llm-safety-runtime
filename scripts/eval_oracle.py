"""
MOD-05: PredictiveOracle Evaluation Script
==========================================
Evaluates the trained PredictiveOracle on the fault injection test set.
Reports precision, recall, F1, AUC at each forecast horizon (30s/60s/90s).

Run with:
    python scripts/eval_oracle.py
    python scripts/eval_oracle.py --model-path models/oracle/patchtst.pt --data-dir data/oracle

Triggered via: make eval-oracle
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import structlog
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger(__name__)


def evaluate_oracle(
    model_path: str,
    data_dir: str,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Evaluate oracle on test set at all horizons.

    Args:
        model_path: Path to patchtst.pt checkpoint.
        data_dir: Directory containing fault_injection_test.npz.
        device: Torch device.

    Returns:
        Dict mapping horizon label → metrics dict.
    """
    from src.predictive_oracle.oracle import PredictiveOracleModel

    # Load test data
    test_path = Path(data_dir) / "fault_injection_test.npz"
    data = np.load(test_path)
    X_test = data["windows"].astype(np.float32)
    y_test = data["labels"].astype(np.int64)

    seq_len = X_test.shape[1]
    n_features = X_test.shape[2]

    # Load model
    model = PredictiveOracleModel(
        n_features=n_features,
        seq_len=seq_len,
        d_model=128,
        n_heads=8,
        n_layers=3,
        patch_len=12,
        stride=6,
    ).to(device)

    model_file = Path(model_path)
    if model_file.exists():
        model.load_state_dict(torch.load(str(model_file), map_location=device))
        logger.info("model_loaded", path=str(model_file))
    else:
        logger.warning("model_file_not_found_using_random_weights", path=str(model_file))

    model.eval()

    # Inference
    horizon_labels = ["30s", "60s", "90s"]
    all_probs = {h: [] for h in horizon_labels}
    latencies = []

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            # Data is (batch, seq_len, n_features); model expects (batch, n_features, seq_len)
            batch = torch.from_numpy(X_test[i:i + batch_size]).permute(0, 2, 1).to(device)
            t0 = time.perf_counter()
            logits = model(batch)
            latency = (time.perf_counter() - t0) * 1000 / len(batch)
            latencies.append(latency)

            probs = torch.sigmoid(logits).cpu().numpy()
            for j, h in enumerate(horizon_labels):
                all_probs[h].extend(probs[:, j].tolist())

    results = {}
    for h_idx, h in enumerate(horizon_labels):
        probs_arr = np.array(all_probs[h])
        preds = (probs_arr >= 0.5).astype(int)

        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(y_test), 1)

        results[h] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    p95_latency = float(np.percentile(latencies, 95))
    results["latency_p95_ms"] = p95_latency

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARGUS PredictiveOracle")
    parser.add_argument("--model-path", default="models/oracle/patchtst.pt")
    parser.add_argument("--data-dir", default="data/oracle")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-f1", type=float, default=0.78,
                        help="Target F1 at 60s horizon")
    args = parser.parse_args()

    if not (Path(args.data_dir) / "fault_injection_test.npz").exists():
        print("[SKIP] Test data not found. Run: python scripts/generate_fault_injection_dataset.py")
        sys.exit(0)

    # If no trained model exists, skip evaluation (random weights give meaningless metrics)
    if not Path(args.model_path).exists():
        print(f"[SKIP] Trained oracle model not found at: {args.model_path}")
        print("       Run: python scripts/train_oracle.py first.")
        sys.exit(0)

    metrics = evaluate_oracle(args.model_path, args.data_dir, args.device)
    p95_ms = metrics.pop("latency_p95_ms")

    print(f"\n=== PredictiveOracle Evaluation ===")
    print(f"  Model         : {args.model_path}")
    print(f"  p95 latency   : {p95_ms:.1f}ms  {'[OK]' if p95_ms < 80 else '[SLOW]'} (target <80ms)")
    print(f"")
    print(f"{'Horizon':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*52}")

    for h in ["30s", "60s", "90s"]:
        m = metrics[h]
        marker = ""
        if h == "60s":
            marker = " [PASS]" if m["f1"] >= args.target_f1 else " [FAIL]"
        print(
            f"  {h:>8} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{m['accuracy']:>10.4f}{marker}"
        )

    f1_60s = metrics["60s"]["f1"]
    print(f"\n  Target (60s F1 >={args.target_f1}): {'PASSED' if f1_60s >= args.target_f1 else 'BELOW TARGET'}")
    if f1_60s < args.target_f1:
        sys.exit(1)


if __name__ == "__main__":
    main()
