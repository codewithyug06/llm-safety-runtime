"""
MOD-05: PredictiveOracle — Temporal Failure Forecasting
=======================================================
A temporal transformer (PatchTST architecture) trained on multivariate agent
telemetry time-series to predict multi-agent system failure events 30–90 seconds
in advance, with calibrated uncertainty via conformal prediction.

Input features per timestep:
    [safety_score, latency_ms, token_entropy, attention_entropy,
     kl_divergence, probe_p_hallucination, probe_p_jailbreak,
     queue_depth, error_rate]

Output:
    failure_probability: float in [0, 1]
    uncertainty_interval: (lower, upper) from conformal prediction
    horizon_seconds: which horizon this prediction is for

References:
    - Nie et al. "A Time Series is Worth 64 Words: PatchTST" (ICLR 2023)
    - Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction" (2021)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import structlog

logger = structlog.get_logger(__name__)


# ── Feature definitions ───────────────────────────────────────────────────────

TELEMETRY_FEATURES = [
    "safety_score",
    "latency_ms",
    "token_entropy",
    "attention_entropy",
    "kl_divergence",
    "probe_p_hallucination",
    "probe_p_jailbreak",
    "queue_depth",
    "error_rate",
]
N_FEATURES = len(TELEMETRY_FEATURES)  # 9


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TelemetryWindow:
    """A sliding window of agent telemetry for the Oracle's input.

    Args:
        agent_id: The agent being monitored.
        features: Numpy array of shape (window_len, N_FEATURES).
        timestamps_ns: Timestamps for each timestep.
        window_len: Number of timesteps in window (e.g., 60 = 60 seconds at 1Hz).
    """
    agent_id: str
    features: np.ndarray  # (window_len, N_FEATURES)
    timestamps_ns: np.ndarray
    window_len: int = 60

    def to_tensor(self, device: str = "cpu") -> Tensor:
        """Convert to PyTorch tensor for model input.

        Returns:
            Tensor of shape (1, N_FEATURES, window_len) — batch=1, channels=features.
        """
        t = torch.tensor(self.features, dtype=torch.float32)
        return t.transpose(0, 1).unsqueeze(0).to(device)  # (1, N_FEATURES, window_len)


@dataclass
class OraclePrediction:
    """Output of the PredictiveOracle for a single agent.

    Args:
        agent_id: The agent this prediction is for.
        failure_probability: Point estimate of failure in [0, 1].
        lower_bound: Conformal prediction lower bound.
        upper_bound: Conformal prediction upper bound.
        horizon_seconds: How far ahead this prediction looks.
        confidence_level: Conformal prediction coverage (e.g., 0.90).
        alert_triggered: Whether this exceeds the alert threshold.
        latency_ms: Time to compute this prediction.
    """
    agent_id: str
    failure_probability: float
    lower_bound: float
    upper_bound: float
    horizon_seconds: int
    confidence_level: float = 0.90
    alert_triggered: bool = False
    latency_ms: float = 0.0


# ── PatchTST-inspired model ───────────────────────────────────────────────────

class TelemetryPatchEmbedding(nn.Module):
    """Patch-based embedding for multivariate telemetry time-series.

    Splits each feature channel into overlapping patches, projects each patch
    to d_model, and adds learnable per-channel and per-position embeddings.
    This is the PatchTST approach — treating time-series patches like ViT
    treats image patches, with per-channel identity preserved.

    Args:
        n_features: Number of input features (N_FEATURES).
        seq_len: Input sequence length in timesteps.
        patch_len: Length of each patch in timesteps.
        stride: Stride between consecutive patches.
        d_model: Embedding dimension.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Shared patch projection: patch_len → d_model
        self.proj = nn.Linear(patch_len, d_model)
        # Learnable channel embedding (one per feature)
        self.channel_embed = nn.Embedding(n_features, d_model)
        # Learnable position embedding (one per patch position)
        self.pos_embed = nn.Embedding(self.n_patches, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Embed time-series into patch tokens.

        Args:
            x: Input tensor of shape (batch, n_features, seq_len).

        Returns:
            Token embeddings of shape (batch, n_features * n_patches, d_model).
        """
        B, C, T = x.shape
        # Unfold: (B, C, n_patches, patch_len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # Project patches: (B, C, n_patches, d_model)
        tok = self.proj(patches)
        # Add position embedding
        pos_idx = torch.arange(self.n_patches, device=x.device)
        tok = tok + self.pos_embed(pos_idx).unsqueeze(0).unsqueeze(0)
        # Add channel embedding
        ch_idx = torch.arange(C, device=x.device)
        tok = tok + self.channel_embed(ch_idx).unsqueeze(0).unsqueeze(2)
        # Flatten channel × patch dims: (B, C * n_patches, d_model)
        tok = tok.reshape(B, C * self.n_patches, -1)
        return self.norm(tok)


class PredictiveOracleModel(nn.Module):
    """Temporal transformer for multi-agent failure prediction.

    Architecture:
        Input → TelemetryPatchEmbedding → CLS prepend → Transformer Encoder
              → CLS output → Independent MLP head per horizon

    Each forecast horizon (30s, 60s, 90s) has its own MLP head, allowing
    independent calibration per horizon.

    Args:
        n_features: Number of telemetry features (default: 9).
        seq_len: Input sequence length in timesteps (default: 60).
        patch_len: Patch length for embedding (default: 12).
        stride: Patch stride (default: 6).
        d_model: Transformer model dimension (default: 128).
        n_heads: Number of attention heads (default: 8).
        n_layers: Number of transformer encoder layers (default: 3).
        dropout: Dropout rate (default: 0.1).
        n_horizons: Number of forecast horizons (default: 3 → 30s/60s/90s).

    Example:
        model = PredictiveOracleModel()
        window = TelemetryWindow(...)
        logits = model(window.to_tensor())  # Shape: (1, n_horizons)
        probs = logits.sigmoid()
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        seq_len: int = 60,
        patch_len: int = 12,
        stride: int = 6,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        n_horizons: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_horizons = n_horizons

        # Patch embedding: projects (B, n_features, seq_len) → (B, tokens, d_model)
        self.patch_embed = TelemetryPatchEmbedding(
            n_features=n_features,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )

        # Transformer encoder (pre-norm for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # CLS-style aggregation token (ViT-style)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Independent MLP head per horizon
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_features, seq_len).

        Returns:
            Logits tensor of shape (batch, n_horizons) — apply sigmoid for probs.
        """
        B = x.shape[0]
        # Patch embed: (B, n_tokens, d_model)
        tokens = self.patch_embed(x)
        # Prepend CLS token: (B, 1 + n_tokens, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        # Transformer: (B, 1 + n_tokens, d_model)
        encoded = self.transformer(tokens)
        # Extract CLS output: (B, d_model)
        cls_out = encoded[:, 0, :]
        # Independent head per horizon: list of (B, 1) → (B, n_horizons)
        logits = torch.cat([head(cls_out) for head in self.heads], dim=1)
        return logits  # (B, n_horizons)


# ── Conformal Prediction Calibration ─────────────────────────────────────────

class ConformalCalibrator:
    """Calibrates model scores with distribution-free conformal prediction.

    Computes non-conformity scores on a calibration set and produces
    prediction intervals with guaranteed coverage at the specified level.

    Args:
        coverage: Target coverage probability (e.g., 0.90).
        horizon_idx: Which horizon this calibrator is for.

    Example:
        calibrator = ConformalCalibrator(coverage=0.90)
        calibrator.fit(cal_probs, cal_labels)
        lower, upper = calibrator.predict_interval(test_prob)
    """

    def __init__(self, coverage: float = 0.90, horizon_idx: int = 1) -> None:
        self.coverage = coverage
        self.horizon_idx = horizon_idx
        self._q_hat: Optional[float] = None

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> None:
        """Compute non-conformity scores on calibration set.

        Args:
            cal_probs: Model probability predictions on calibration set.
            cal_labels: True binary labels (0 = no failure, 1 = failure).
        """
        # Non-conformity score: 1 - p(true class)
        scores = np.where(cal_labels == 1, 1 - cal_probs, cal_probs)

        n = len(scores)
        q_level = math.ceil((n + 1) * self.coverage) / n
        self._q_hat = float(np.quantile(scores, q_level))

        logger.info(
            "conformal_calibrated",
            q_hat=self._q_hat,
            coverage=self.coverage,
            n_cal=n,
        )

    def predict_interval(self, prob: float) -> Tuple[float, float]:
        """Compute conformal prediction interval for a probability estimate.

        Args:
            prob: Model's point probability estimate.

        Returns:
            Tuple (lower_bound, upper_bound) — guaranteed to contain true label
            with probability >= self.coverage on exchangeable data.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        if self._q_hat is None:
            raise RuntimeError("Calibrator must be fitted before predicting.")

        lower = max(0.0, prob - self._q_hat)
        upper = min(1.0, prob + self._q_hat)
        return lower, upper


# ── Oracle: Top-level interface ───────────────────────────────────────────────

class PredictiveOracle:
    """Top-level interface for MOD-05.

    Combines PredictiveOracleModel + ConformalCalibrator into a single
    production-ready interface for predicting agent failure events.

    Args:
        model: Trained PredictiveOracleModel.
        calibrators: Per-horizon ConformalCalibrator instances.
        alert_threshold: Failure probability above which to trigger alert.
        horizons_seconds: Forecast horizons (default: [30, 60, 90]).
        device: Compute device.

    Example:
        oracle = PredictiveOracle.load("models/oracle_v3")
        window = telemetry_collector.get_window("agent-001")
        prediction = oracle.predict(window)
        if prediction.alert_triggered:
            remediator.run(state)
    """

    HORIZONS = [30, 60, 90]

    def __init__(
        self,
        model: PredictiveOracleModel,
        calibrators: Dict[int, ConformalCalibrator],
        alert_threshold: float = 0.75,
        device: str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._model.eval()
        self._calibrators = calibrators
        self._threshold = alert_threshold
        self._device = device

    @torch.no_grad()
    def predict(
        self,
        window: TelemetryWindow,
        primary_horizon: int = 60,
    ) -> OraclePrediction:
        """Generate failure prediction for a telemetry window.

        Args:
            window: Sliding window of agent telemetry data.
            primary_horizon: The main horizon to report (30, 60, or 90).

        Returns:
            OraclePrediction with failure probability and uncertainty bounds.
        """
        import time as _time
        t0 = _time.monotonic_ns()

        x = window.to_tensor(self._device)
        logits = self._model(x)  # (1, n_horizons)
        probs = logits.sigmoid().squeeze(0).cpu().numpy()

        horizon_idx = self.HORIZONS.index(primary_horizon) if primary_horizon in self.HORIZONS else 1
        prob = float(probs[horizon_idx])

        # Apply conformal calibration
        calibrator = self._calibrators.get(primary_horizon)
        if calibrator:
            lower, upper = calibrator.predict_interval(prob)
        else:
            lower = max(0.0, prob - 0.15)
            upper = min(1.0, prob + 0.15)

        latency_ms = (_time.monotonic_ns() - t0) / 1e6

        return OraclePrediction(
            agent_id=window.agent_id,
            failure_probability=prob,
            lower_bound=lower,
            upper_bound=upper,
            horizon_seconds=primary_horizon,
            alert_triggered=prob >= self._threshold,
            latency_ms=latency_ms,
        )

    def predict_all_horizons(
        self,
        window: TelemetryWindow,
    ) -> List[OraclePrediction]:
        """Generate predictions for all three horizons.

        Args:
            window: Telemetry window.

        Returns:
            List of OraclePrediction, one per horizon.
        """
        return [self.predict(window, h) for h in self.HORIZONS]

    @classmethod
    def load(cls, model_path: str, device: str = "cpu") -> "PredictiveOracle":
        """Load a trained oracle from disk.

        Args:
            model_path: Path to saved model checkpoint directory.
            device: Compute device.

        Returns:
            Initialized PredictiveOracle ready for inference.
        """
        import os

        checkpoint = torch.load(
            os.path.join(model_path, "model.pt"),
            map_location=device,
        )
        model = PredictiveOracleModel(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])

        calibrators = {}
        for horizon in cls.HORIZONS:
            cal_path = os.path.join(model_path, f"calibrator_{horizon}s.npy")
            if os.path.exists(cal_path):
                cal = ConformalCalibrator(coverage=0.90, horizon_idx=cls.HORIZONS.index(horizon))
                cal._q_hat = float(np.load(cal_path))
                calibrators[horizon] = cal

        return cls(model=model, calibrators=calibrators, device=device)
