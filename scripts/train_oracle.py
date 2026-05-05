"""
MOD-05: PredictiveOracle Training Script
==========================================
Trains the PatchTST-based PredictiveOracle on the fault injection dataset,
calibrates conformal prediction intervals, evaluates on test set, and
registers the model + calibrator to MLflow.

Run with:
    python scripts/train_oracle.py
    python scripts/train_oracle.py --data-dir data/oracle --epochs 50 --lr 1e-3
    python scripts/train_oracle.py --dry-run

Triggered via: make train-oracle
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_oracle_config
from src.predictive_oracle.oracle import ConformalCalibrator, PredictiveOracleModel

logger = structlog.get_logger(__name__)


def load_split(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a .npz split file.

    Args:
        data_dir: Directory containing fault_injection_*.npz files.
        split: One of 'train', 'val', 'test'.

    Returns:
        Tuple of (windows, labels) as numpy arrays.

    Raises:
        FileNotFoundError: If the split file is missing.
    """
    path = data_dir / f"fault_injection_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}\n"
            f"Run: python scripts/generate_fault_injection_dataset.py --output-dir {data_dir}"
        )
    data = np.load(path)
    return data["windows"].astype(np.float32), data["labels"].astype(np.int64)


def build_dataloader(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader.

    Args:
        windows: Shape (N, seq_len, n_features).
        labels: Shape (N,).
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        PyTorch DataLoader.
    """
    dataset = TensorDataset(
        torch.from_numpy(windows),
        torch.from_numpy(labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(
    model: PredictiveOracleModel,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: str,
) -> float:
    """Run one training epoch.

    Args:
        model: PredictiveOracleModel in train mode.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        criterion: Loss function (BCEWithLogitsLoss).
        device: Torch device.

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for windows_batch, labels_batch in loader:
        windows_batch = windows_batch.to(device)
        labels_batch = labels_batch.float().to(device)

        optimizer.zero_grad()
        logits = model(windows_batch)  # (batch, n_horizons)

        # Use 60s horizon (index 1 in [30, 60, 90])
        loss = criterion(logits[:, 1], labels_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: PredictiveOracleModel,
    loader: DataLoader,
    device: str,
    horizon_idx: int = 1,  # 60s horizon
) -> Dict[str, float]:
    """Evaluate model on a DataLoader.

    Args:
        model: PredictiveOracleModel in eval mode.
        loader: DataLoader.
        device: Torch device.
        horizon_idx: Which horizon to evaluate (0=30s, 1=60s, 2=90s).

    Returns:
        Dict with loss, accuracy, precision, recall, f1.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for windows_batch, labels_batch in loader:
        windows_batch = windows_batch.to(device)
        labels_batch = labels_batch.float().to(device)

        logits = model(windows_batch)
        loss = criterion(logits[:, horizon_idx], labels_batch)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits[:, horizon_idx]).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels_batch.cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    tp = int(((preds == 1) & (all_labels == 1)).sum())
    fp = int(((preds == 1) & (all_labels == 0)).sum())
    fn = int(((preds == 0) & (all_labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = float((preds == all_labels).mean())

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ARGUS PredictiveOracle")
    parser.add_argument("--data-dir", default="data/oracle")
    parser.add_argument("--output-dir", default="models/oracle")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mlflow-experiment", default="argus/predictive_oracle")
    parser.add_argument("--wandb-project", default="argus-oracle")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "train_oracle_start",
        data_dir=str(data_dir),
        epochs=args.epochs,
        device=args.device,
    )

    # Load data splits
    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")
    X_test, y_test = load_split(data_dir, "test")

    logger.info(
        "data_loaded",
        train=len(X_train),
        val=len(X_val),
        test=len(X_test),
        window_shape=X_train.shape[1:],
    )

    if args.dry_run:
        print(f"\n=== Dry Run: Validated ===")
        print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
        print(f"  Window shape: {X_train.shape[1:]}")
        print(f"  Device: {args.device}")
        print("\nDry run complete.")
        return

    # Load config (for model architecture)
    try:
        cfg = load_oracle_config()
        hidden_dim = cfg.model.hidden_dim
        num_heads = cfg.model.num_heads
        num_layers = cfg.model.num_layers
        patch_len = cfg.model.patch_len
        stride = cfg.model.stride
        num_features = cfg.model.num_features
        forecast_horizons = cfg.model.forecast_horizons
    except Exception:
        # Fallback defaults matching predictive_oracle.yaml
        hidden_dim, num_heads, num_layers = 128, 8, 3
        patch_len, stride, num_features = 12, 6, 9
        forecast_horizons = [30, 60, 90]

    seq_len = X_train.shape[1]
    n_horizons = len(forecast_horizons)

    # Build model — PredictiveOracleModel expects (batch, n_features, seq_len)
    # so we transpose data after loading: (N, seq_len, n_features) → (N, n_features, seq_len)
    X_train = X_train.transpose(0, 2, 1)
    X_val   = X_val.transpose(0, 2, 1)
    X_test  = X_test.transpose(0, 2, 1)

    model = PredictiveOracleModel(
        n_features=num_features,
        seq_len=seq_len,
        d_model=hidden_dim,
        n_heads=num_heads,
        n_layers=num_layers,
        patch_len=patch_len,
        stride=stride,
        n_horizons=n_horizons,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model_built", params=n_params, device=args.device)

    # DataLoaders
    train_loader = build_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = build_dataloader(X_test, y_test, args.batch_size, shuffle=False)

    # Optimizer + scheduler + loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    # MLflow setup
    try:
        import mlflow
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow_run = mlflow.start_run(run_name="patchtst_oracle")
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "n_params": n_params,
        })
        _mlflow_active = True
    except Exception:
        _mlflow_active = False

    # W&B setup
    if not args.skip_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except Exception:
            pass

    # Training loop with early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_path = output_dir / "patchtst_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = evaluate(model, val_loader, args.device)
        scheduler.step()
        epoch_time = time.perf_counter() - t0

        logger.info(
            "epoch_complete",
            epoch=epoch,
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_f1=f"{val_metrics['f1']:.4f}",
            val_precision=f"{val_metrics['precision']:.4f}",
            val_recall=f"{val_metrics['recall']:.4f}",
            epoch_time_s=f"{epoch_time:.1f}",
        )

        if _mlflow_active:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                },
                step=epoch,
            )

        # Early stopping on val F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("best_model_saved", val_f1=f"{best_val_f1:.4f}", path=str(best_model_path))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("early_stopping", epoch=epoch, patience=args.patience)
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    logger.info("best_model_loaded", path=str(best_model_path))

    # Calibrate conformal predictor on val set
    logger.info("calibrating_conformal_predictor")
    calibrator = ConformalCalibrator(coverage=0.9)

    # Collect val probabilities for calibration
    model.eval()
    val_probs = []
    with torch.no_grad():
        for windows_batch, _ in val_loader:
            logits = model(windows_batch.to(args.device))
            probs = torch.sigmoid(logits[:, 1]).cpu().numpy()  # 60s horizon
            val_probs.extend(probs.tolist())

    calibrator.fit(np.array(val_probs), y_val)
    calibrator_path = output_dir / "calibrator.pkl"
    with calibrator_path.open("wb") as f:
        pickle.dump(calibrator, f)
    logger.info("calibrator_saved", path=str(calibrator_path))

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, args.device)
    logger.info("test_evaluation_complete", **{k: f"{v:.4f}" for k, v in test_metrics.items()})

    # Save final model
    final_model_path = output_dir / "patchtst.pt"
    torch.save(model.state_dict(), final_model_path)

    if _mlflow_active:
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in test_metrics.items()},
            step=args.epochs,
        )
        mlflow.log_artifact(str(final_model_path))
        mlflow.log_artifact(str(calibrator_path))
        mlflow.end_run()

    try:
        import wandb
        wandb.finish()
    except Exception:
        pass

    # Print summary
    print(f"\n=== PredictiveOracle Training Complete ===")
    print(f"  Epochs        : {args.epochs} (best at patience stop)")
    print(f"  Best val F1   : {best_val_f1:.4f}")
    print(f"")
    print(f"Test metrics (60s horizon):")
    for k, v in test_metrics.items():
        marker = " [OK]" if k == "f1" and v >= 0.78 else (" [LOW]" if k == "f1" else "")
        print(f"  {k:12}: {v:.4f}{marker}")
    print(f"")
    print(f"  Target F1     : >=0.78 at 60s  {'[PASSED]' if test_metrics['f1'] >= 0.78 else '[BELOW TARGET]'}")
    print(f"  Model saved   : {final_model_path}")
    print(f"  Calibrator    : {calibrator_path}")


if __name__ == "__main__":
    main()
