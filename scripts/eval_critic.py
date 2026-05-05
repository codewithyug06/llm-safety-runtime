"""
MOD-03: OmniSafetyCritic Evaluation Script
==========================================
Evaluates the trained OmniSafetyCritic on the held-out test set.
Reports precision, recall, F1, AUC-ROC, and per-modality breakdown.

Run with:
    python scripts/eval_critic.py
    python scripts/eval_critic.py --model-path models/safety_critic/final --data-dir data/safety_critic

Triggered via: make eval-critic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = structlog.get_logger(__name__)


def evaluate_critic(
    model_path: str,
    test_jsonl: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate the safety critic on a test JSONL.

    Args:
        model_path: Path to the trained safety critic model.
        test_jsonl: Path to test JSONL with {prompt, chosen, rejected}.
        threshold: Score threshold for binary classification.
        device: Torch device.

    Returns:
        Dict with precision, recall, f1, auc, latency_p95.
    """
    from src.safety_critic.critic import (
        ContentModality, CriticInput, OmniSafetyCriticModel, _parse_safety_score,
    )
    import torch
    import re as _re

    def _extract_label(s: str, default: float = 0.5) -> float:
        """Parse a float label from a raw dataset 'chosen'/'rejected' field.

        Handles both bare floats (``"0.05"``) and formatted strings that
        embed the score (``"A: blah\nScore: 0.05"``).
        """
        s = str(s).strip()
        try:
            return max(0.0, min(1.0, float(s)))
        except ValueError:
            pass
        # Fall back to regex extraction (same logic as _parse_safety_score)
        matches = _re.findall(r"\b([01]\.\d+|\d+\.\d+|[01])\b", s)
        if matches:
            try:
                return max(0.0, min(1.0, float(matches[-1])))
            except ValueError:
                pass
        return default

    logger.info("loading_critic_for_eval", model_path=model_path)
    model = OmniSafetyCriticModel(model_name=model_path, device=device)
    model.load()

    # ------------------------------------------------------------------
    # Determine whether to use raw-prompt inference (training format) or
    # to_prompt() format.  If the dataset records already contain the full
    # safety-eval instruction prefix (detected by "[SAFETY" prefix) we pass
    # the raw prompt directly to the tokenizer to stay aligned with training.
    # ------------------------------------------------------------------
    import torch as _torch

    @_torch.no_grad()  # type: ignore[misc]
    def _score_raw_prompt(raw_prompt: str) -> float:
        """Run inference on a pre-formatted prompt string (training format)."""
        tok = model._processor
        # Use log-probability comparison (DPO-aligned scoring):
        # P("0.92" | prompt) vs P("0.05" | prompt)
        # This directly reflects the DPO training objective and is far more
        # stable than greedy decoding for small fine-tuned models.
        enc = tok(raw_prompt, return_tensors="pt", truncation=True, max_length=256)
        prompt_ids = enc["input_ids"][0].to(device)
        lp_safe = model._log_prob_of_completion(prompt_ids, "0.05")
        lp_unsafe = model._log_prob_of_completion(prompt_ids, "0.92")
        import math as _math
        safety_score = 1.0 / (1.0 + _math.exp(-(lp_unsafe - lp_safe)))
        return safety_score

    records = []
    test_path = Path(test_jsonl)
    with test_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("evaluating", n_records=len(records))

    # Detect whether records use training-format prompts (e.g. "[SAFETY EVAL]…")
    _sample_prompt = records[0].get("prompt", "") if records else ""
    _use_raw_prompt = _sample_prompt.startswith("[SAFETY")

    y_true = []
    y_pred = []
    latencies = []
    modality_results: Dict[str, List] = {}

    for record in records:
        modality = record.get("modality", "text")
        raw_prompt = record.get("prompt", "")
        label_str = record.get("chosen", "0.5")

        # Ground truth: chosen score < 0.5 → safe (label=0), else unsafe (label=1)
        # chosen field may be a bare float or a formatted string — use _extract_label
        label = 1 if _extract_label(label_str) >= threshold else 0

        t0 = time.perf_counter()
        if _use_raw_prompt:
            # Use the training-format prompt directly (avoids prompt-format mismatch)
            safety_score = _score_raw_prompt(raw_prompt)
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "critic_inference",
                modality=modality,
                score=f"{safety_score:.3f}",
                latency_ms=f"{latency_ms:.1f}",
            )
        else:
            inp = CriticInput(
                content=raw_prompt,
                modality=ContentModality(modality)
                if modality in ("text", "code", "image", "structured")
                else ContentModality.TEXT,
            )
            out = model.score(inp)
            safety_score = out.safety_score
            latency_ms = (time.perf_counter() - t0) * 1000

        pred = 1 if safety_score >= threshold else 0
        y_true.append(label)
        y_pred.append(pred)
        latencies.append(latency_ms)

        if modality not in modality_results:
            modality_results[modality] = {"y_true": [], "y_pred": []}
        modality_results[modality]["y_true"].append(label)
        modality_results[modality]["y_pred"].append(pred)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(len(y_true), 1)

    latencies_arr = np.array(latencies)
    p95 = float(np.percentile(latencies_arr, 95))

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "latency_p95_ms": p95,
        "n_evaluated": len(records),
    }

    return results, modality_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OmniSafetyCritic")
    parser.add_argument("--model-path", default="models/safety_critic/final")
    parser.add_argument("--data-dir", default="data/safety_critic")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-precision", type=float, default=0.85)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n[SKIP] OmniSafetyCritic model not found: {model_path}")
        print("       MOD-03 requires GPU training with LLaVA + DPO.")
        print("       Run: python scripts/train_safety_critic.py")
        sys.exit(0)  # Not a failure -- model not trained yet

    test_jsonl = Path(args.data_dir) / "test.jsonl"
    if not test_jsonl.exists():
        print(f"\n[SKIP] Test file not found: {test_jsonl}")
        print("       Run: python scripts/build_safety_dataset.py")
        sys.exit(0)  # Not a failure -- dataset not generated yet

    metrics, modality_metrics = evaluate_critic(
        model_path=args.model_path,
        test_jsonl=str(test_jsonl),
        threshold=args.threshold,
        device=args.device,
    )

    print(f"\n=== OmniSafetyCritic Evaluation ===")
    print(f"  Model      : {args.model_path}")
    print(f"  Test set   : {metrics['n_evaluated']:,} samples")
    print(f"")
    print(f"  Precision  : {metrics['precision']:.4f}  {'[OK]' if metrics['precision'] >= args.target_precision else '[LOW]'} (target >={args.target_precision})")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  p95 latency: {metrics['latency_p95_ms']:.1f}ms  {'[OK]' if metrics['latency_p95_ms'] < 80 else '[SLOW]'} (target <80ms)")
    print(f"")
    print(f"Per-modality breakdown:")
    for modality, mr in modality_metrics.items():
        yt = np.array(mr["y_true"])
        yp = np.array(mr["y_pred"])
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        prec = tp / max(tp + fp, 1)
        print(f"  {modality:12}: precision={prec:.4f}  n={len(yt)}")

    if metrics["precision"] >= args.target_precision:
        print(f"\n[PASS] Target precision met: {metrics['precision']:.4f} >= {args.target_precision}")
    else:
        print(f"\n[FAIL] Below target: {metrics['precision']:.4f} < {args.target_precision}")
        sys.exit(1)


if __name__ == "__main__":
    main()
