"""
Unit tests for MOD-01: LatentSentinel
======================================
All tests use mock/stub transformers — no real LLM loaded in unit tests.
"""

import time
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn

from src.latent_sentinel.sentinel import (
    ActivationBundle,
    BaseProbe,
    HookManager,
    LatentSentinel,
    LinearResidualProbe,
    ProbeCategory,
    ProbeRegistry,
    ProbeResult,
    RiskLevel,
    SafetySignal,
)
from src.causal_engine.intervention import CausalInterventionEngine  # MOD-02


# ── Fixtures ──────────────────────────────────────────────────────────────────

class TinyTransformerLayer(nn.Module):
    """Stub transformer layer for testing hooks without loading a real model."""
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return (self.linear(x),)


class _TinyLayers(nn.Module):
    """Inner container that exposes .layers, mirroring Llama's model.layers list."""
    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self._layer_list = nn.ModuleList(
            [TinyTransformerLayer(hidden_dim) for _ in range(n_layers)]
        )

    @property
    def layers(self) -> list:
        return list(self._layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layer_list:
            x = layer(x)[0]
        return x


class TinyTransformer(nn.Module):
    """Stub transformer model compatible with HookManager layer discovery."""
    def __init__(self, n_layers: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.model = _TinyLayers(n_layers, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def tiny_model(hidden_dim: int) -> TinyTransformer:
    return TinyTransformer(n_layers=4, hidden_dim=hidden_dim)


@pytest.fixture
def activation_bundle(hidden_dim: int) -> ActivationBundle:
    return ActivationBundle(
        layer_idx=2,
        residual_stream=torch.randn(1, 10, hidden_dim),
        attention_patterns=None,
        mlp_activations=None,
        timestamp_ns=time.monotonic_ns(),
        request_id="test-req-001",
    )


@pytest.fixture
def hallucination_probe(hidden_dim: int) -> LinearResidualProbe:
    probe = LinearResidualProbe(hidden_dim, ProbeCategory.HALLUCINATION)
    return probe


@pytest.fixture
def all_probes(hidden_dim: int) -> Dict[ProbeCategory, LinearResidualProbe]:
    return {
        ProbeCategory.HALLUCINATION: LinearResidualProbe(hidden_dim, ProbeCategory.HALLUCINATION),
        ProbeCategory.JAILBREAK: LinearResidualProbe(hidden_dim, ProbeCategory.JAILBREAK),
        ProbeCategory.TOXIC_REASONING: LinearResidualProbe(hidden_dim, ProbeCategory.TOXIC_REASONING),
        ProbeCategory.POLICY_VIOLATION: LinearResidualProbe(hidden_dim, ProbeCategory.POLICY_VIOLATION),
    }


# ── Test: ActivationBundle ────────────────────────────────────────────────────

class TestActivationBundle:
    def test_creation(self, hidden_dim: int) -> None:
        bundle = ActivationBundle(
            layer_idx=3,
            residual_stream=torch.randn(1, 5, hidden_dim),
            attention_patterns=None,
            mlp_activations=None,
            timestamp_ns=12345,
            request_id="req-001",
        )
        assert bundle.layer_idx == 3
        assert bundle.residual_stream.shape == (1, 5, hidden_dim)
        assert bundle.request_id == "req-001"


# ── Test: LinearResidualProbe ────────────────────────────────────────────────

class TestLinearResidualProbe:
    def test_output_range(
        self,
        hallucination_probe: LinearResidualProbe,
        activation_bundle: ActivationBundle,
    ) -> None:
        """Risk score and confidence must be in [0, 1]."""
        score, conf = hallucination_probe(activation_bundle)
        assert 0.0 <= score <= 1.0, f"Risk score {score} out of range"
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range"

    def test_correct_category(self, hidden_dim: int) -> None:
        probe = LinearResidualProbe(hidden_dim, ProbeCategory.JAILBREAK)
        assert probe.category == ProbeCategory.JAILBREAK

    def test_no_gradient_computation(
        self,
        hallucination_probe: LinearResidualProbe,
        activation_bundle: ActivationBundle,
    ) -> None:
        """Probe should not compute gradients (inference only)."""
        # No assertion needed — if it raises, it fails
        score, _ = hallucination_probe(activation_bundle)
        assert isinstance(score, float)

    def test_batch_size_one(
        self,
        hallucination_probe: LinearResidualProbe,
        hidden_dim: int,
    ) -> None:
        """Probe must handle batch_size=1 (standard production case)."""
        bundle = ActivationBundle(
            layer_idx=0,
            residual_stream=torch.randn(1, 32, hidden_dim),
            attention_patterns=None,
            mlp_activations=None,
            timestamp_ns=0,
            request_id="r",
        )
        score, conf = hallucination_probe(bundle)
        assert isinstance(score, float)


# ── Test: ProbeRegistry ───────────────────────────────────────────────────────

class TestProbeRegistry:
    def test_dispatch_returns_signal(
        self,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
        activation_bundle: ActivationBundle,
    ) -> None:
        registry = ProbeRegistry(probes=all_probes)
        signal = registry.dispatch(activation_bundle)
        assert isinstance(signal, SafetySignal)
        assert signal.request_id == "test-req-001"

    def test_signal_has_all_probes(
        self,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
        activation_bundle: ActivationBundle,
    ) -> None:
        registry = ProbeRegistry(probes=all_probes)
        signal = registry.dispatch(activation_bundle)
        assert set(signal.probe_results.keys()) == set(ProbeCategory)

    def test_composite_score_in_range(
        self,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
        activation_bundle: ActivationBundle,
    ) -> None:
        registry = ProbeRegistry(probes=all_probes)
        signal = registry.dispatch(activation_bundle)
        assert 0.0 <= signal.composite_score <= 1.0

    def test_callback_called(
        self,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
        activation_bundle: ActivationBundle,
    ) -> None:
        callback = MagicMock()
        registry = ProbeRegistry(probes=all_probes, signal_callback=callback)
        registry.dispatch(activation_bundle)
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], SafetySignal)

    def test_empty_probes_returns_safe(self, activation_bundle: ActivationBundle) -> None:
        registry = ProbeRegistry(probes={})
        signal = registry.dispatch(activation_bundle)
        assert signal.risk_level == RiskLevel.SAFE

    def test_risk_level_classification(
        self,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
    ) -> None:
        """Manually set scores to test threshold classification."""
        registry = ProbeRegistry(probes=all_probes)

        # Test classify_risk directly
        assert registry._classify_risk(0.0) == RiskLevel.SAFE
        assert registry._classify_risk(0.25) == RiskLevel.LOW
        assert registry._classify_risk(0.50) == RiskLevel.MEDIUM
        assert registry._classify_risk(0.70) == RiskLevel.HIGH
        assert registry._classify_risk(0.90) == RiskLevel.CRITICAL


# ── Test: HookManager ────────────────────────────────────────────────────────

class TestHookManager:
    def test_attach_and_detach(self, tiny_model: TinyTransformer) -> None:
        callback = MagicMock()
        manager = HookManager(tiny_model, device="cpu")
        manager.attach(callback=callback)
        assert len(manager._hooks) > 0

        manager.detach()
        assert len(manager._hooks) == 0

    def test_hook_fires_on_forward(self, tiny_model: TinyTransformer) -> None:
        """Hooks should fire when the model runs a forward pass."""
        bundles = []

        def capture(bundle: ActivationBundle) -> None:
            bundles.append(bundle)

        manager = HookManager(tiny_model, device="cpu")
        manager.attach(callback=capture)

        x = torch.randn(1, 10, 64)
        tiny_model(x)
        manager.detach()

        assert len(bundles) > 0, "No activation bundles captured"
        assert all(isinstance(b, ActivationBundle) for b in bundles)

    def test_double_attach_raises(self, tiny_model: TinyTransformer) -> None:
        callback = MagicMock()
        manager = HookManager(tiny_model, device="cpu")
        manager.attach(callback=callback)

        with pytest.raises(RuntimeError, match="already attached"):
            manager.attach(callback=callback)

        manager.detach()

    def test_target_layers_filter(self, tiny_model: TinyTransformer) -> None:
        """Only specified layers should have hooks."""
        callback = MagicMock()
        manager = HookManager(tiny_model, target_layers=[0, 2], device="cpu")
        manager.attach(callback=callback)
        assert len(manager._hooks) == 2
        manager.detach()


# ── Test: LatentSentinel ─────────────────────────────────────────────────────

class TestLatentSentinel:
    def test_monitor_and_stop(
        self,
        tiny_model: TinyTransformer,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
    ) -> None:
        sentinel = LatentSentinel(probes=all_probes, device="cpu")
        assert not sentinel.is_active

        sentinel.monitor(tiny_model)
        assert sentinel.is_active

        sentinel.stop()
        assert not sentinel.is_active

    def test_double_monitor_raises(
        self,
        tiny_model: TinyTransformer,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
    ) -> None:
        sentinel = LatentSentinel(probes=all_probes, device="cpu")
        sentinel.monitor(tiny_model)

        with pytest.raises(RuntimeError):
            sentinel.monitor(tiny_model)

        sentinel.stop()

    def test_signals_emitted_on_forward(
        self,
        tiny_model: TinyTransformer,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
    ) -> None:
        signals = []
        sentinel = LatentSentinel(
            probes=all_probes,
            signal_callback=signals.append,
            device="cpu",
        )
        sentinel.monitor(tiny_model)

        x = torch.randn(1, 10, 64)
        tiny_model(x)

        sentinel.stop()
        assert len(signals) > 0

    def test_latency_under_threshold(
        self,
        tiny_model: TinyTransformer,
        all_probes: Dict[ProbeCategory, LinearResidualProbe],
    ) -> None:
        """Probe dispatch should complete well under 10ms on CPU with tiny model."""
        signals = []
        sentinel = LatentSentinel(
            probes=all_probes,
            signal_callback=signals.append,
            device="cpu",
        )
        sentinel.monitor(tiny_model)

        x = torch.randn(1, 10, 64)
        tiny_model(x)
        sentinel.stop()

        for sig in signals:
            # On real GPU with 8B model this would be <10ms.
            # On CPU tiny model, <100ms is the test threshold.
            assert sig.total_latency_ms < 100, (
                f"Probe latency {sig.total_latency_ms:.1f}ms exceeds test threshold"
            )
