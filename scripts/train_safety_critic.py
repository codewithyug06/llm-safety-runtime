"""
MOD-03: OmniSafetyCritic Training Entry Point
===============================================
CLI entry point for DPO fine-tuning of LLaVA-1.6 as a multimodal safety critic.
Orchestrates: dataset loading → model init → DPO training → evaluation → MLflow registration.

Run with:
    python scripts/train_safety_critic.py
    python scripts/train_safety_critic.py --data-dir data/safety_critic --epochs 3
    python scripts/train_safety_critic.py --model llava-hf/llava-v1.6-mistral-7b-hf --dry-run

Triggered via: make train-critic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.safety_critic.critic import SafetyCriticDataset
from src.safety_critic.trainer import SafetyCriticTrainer

logger = structlog.get_logger(__name__)


def _load_hf_dataset(jsonl_path: Path, tokenizer: Any, max_length: int) -> Any:
    """Load a JSONL file into a HuggingFace Dataset for TRL.

    Args:
        jsonl_path: Path to JSONL file with {prompt, chosen, rejected}.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length.

    Returns:
        HuggingFace Dataset object.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run: pip install datasets>=2.18.0")

    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train OmniSafetyCritic via DPO on multimodal safety pairs"
    )
    parser.add_argument(
        "--model",
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="HuggingFace base model",
    )
    parser.add_argument("--data-dir", default="data/safety_critic")
    parser.add_argument("--output-dir", default="models/safety_critic")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--mlflow-experiment", default="argus/safety_critic/dpo")
    parser.add_argument("--wandb-project", default="argus-safety-critic")
    parser.add_argument("--registry-name", default="argus-safety-critic")
    parser.add_argument("--hf-token", default="")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable 4-bit QLoRA (NF4). Required for GPUs with <=8 GB VRAM (e.g. RTX 4050).",
    )
    parser.add_argument("--skip-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset only; do not train",
    )
    parser.add_argument("--skip-register", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    # Validate data files exist
    for p in [train_path, val_path]:
        if not p.exists():
            logger.error("data_file_missing", path=str(p))
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                f"Run: python scripts/build_safety_dataset.py --output-dir {args.data_dir}"
            )

    logger.info(
        "train_safety_critic_start",
        model=args.model,
        data_dir=str(data_dir),
        output_dir=args.output_dir,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )

    # Count samples in each split
    def _count_lines(p: Path) -> int:
        return sum(1 for line in p.open() if line.strip())

    n_train = _count_lines(train_path)
    n_val = _count_lines(val_path)
    n_test = _count_lines(test_path) if test_path.exists() else 0

    logger.info(
        "dataset_summary",
        train=n_train,
        val=n_val,
        test=n_test,
    )

    if args.dry_run:
        print("\n=== Dry Run: Config Validated ===")
        print(f"  Model          : {args.model}")
        print(f"  Train samples  : {n_train:,}")
        print(f"  Val samples    : {n_val:,}")
        print(f"  Test samples   : {n_test:,}")
        print(f"  Epochs         : {args.epochs}")
        print(f"  Batch size     : {args.batch_size} × {args.grad_accum} accum = "
              f"{args.batch_size * args.grad_accum} effective")
        print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, beta={args.dpo_beta}")
        print(f"  Output dir     : {args.output_dir}")
        print("\nDry run complete — no training performed.")
        return

    # Build trainer
    trainer = SafetyCriticTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dpo_beta=args.dpo_beta,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        mlflow_experiment=args.mlflow_experiment,
        wandb_project=args.wandb_project,
        load_in_4bit=args.load_in_4bit,
        skip_wandb=args.skip_wandb,
    )

    # Load model (builds LoRA adapters)
    trainer.load_model()

    # Load datasets via HuggingFace Dataset (TRL expects this format)
    logger.info("loading_datasets")
    # We need tokenizer from trainer to load datasets
    tokenizer = trainer._tokenizer

    train_dataset = _load_hf_dataset(train_path, tokenizer, args.max_length)
    val_dataset = _load_hf_dataset(val_path, tokenizer, args.max_length)

    logger.info(
        "datasets_loaded",
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )

    # Train
    logger.info("starting_dpo_training")
    trainer.train(train_dataset=train_dataset, eval_dataset=val_dataset)

    # Evaluate on test set
    if test_path.exists():
        logger.info("evaluating_on_test_set")
        test_dataset = _load_hf_dataset(test_path, tokenizer, args.max_length)
        metrics = trainer.evaluate(test_dataset)
        print(f"\n=== Test Evaluation ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Register to MLflow model registry
    if not args.skip_register:
        logger.info("registering_to_mlflow", registry_name=args.registry_name)
        uri = trainer.register_to_mlflow(model_name=args.registry_name)
        print(f"\nModel registered: {uri}")

    print(f"\n=== Training Complete ===")
    print(f"  Model     : {args.model}")
    print(f"  Output    : {args.output_dir}/final")
    print(f"  MLflow    : {args.mlflow_experiment}")


if __name__ == "__main__":
    main()
