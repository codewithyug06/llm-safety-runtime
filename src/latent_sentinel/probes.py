"""
src/latent_sentinel/probes.py
==============================
Raw-tensor LinearResidualProbe used by eval_probes.py and the synthetic
training pipeline.  This provides a simple activation-in → logit-out interface,
distinct from the ActivationBundle-based probe in sentinel.py which is used at
hook time during live inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearResidualProbe(nn.Module):
    """Lightweight two-layer probe that scores flat activation vectors.

    Designed for post-hoc evaluation: takes mean-pooled residual stream
    activations of shape ``(N, input_dim)`` and returns scalar logits.

    Args:
        input_dim: Dimensionality of the input activation vector.
        hidden_dim: Size of the single hidden layer (default 64).

    Example:
        probe = LinearResidualProbe(input_dim=128)
        activations = torch.randn(32, 128)   # batch of 32 samples
        logits = probe(activations)           # (32,)
        probs  = torch.sigmoid(logits)        # unsafe probability per sample
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of activations.

        Args:
            x: Float tensor of shape ``(N, input_dim)``.

        Returns:
            Float tensor of shape ``(N,)`` — raw logits (apply sigmoid for probs).
        """
        return self.net(x).squeeze(-1)
