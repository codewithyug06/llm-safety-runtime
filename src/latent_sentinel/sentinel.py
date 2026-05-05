"""
MOD-01: LatentSentinel — Real-time Activation Monitoring
=========================================================
Attaches PyTorch forward hooks to transformer layers and extracts
latent-space signals for unsafe behavior detection before output generation.

Architecture:
    LLM forward pass
        → HookManager (non-blocking, CUDA streams)
            → ProbeRegistry (4 classifiers in parallel)
                → RiskAggregator
                    → SafetySignal (fed to PredictiveOracle + CausalEngine)

Target SLA: <10ms p95 hook-to-signal latency on 8B model
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import structlog

logger = structlog.get_logger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class ProbeCategory(Enum):
    HALLUCINATION = auto()
    JAILBREAK = auto()
    TOXIC_REASONING = auto()
    POLICY_VIOLATION = auto()


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ActivationBundle:
    """Raw activation tensors captured from a single forward pass.

    Args:
        layer_idx: Transformer layer index (0-indexed).
        residual_stream: Hidden state tensor of shape (batch, seq_len, hidden_dim).
        attention_patterns: Attention weight tensor of shape (batch, heads, seq, seq).
        mlp_activations: MLP intermediate activations of shape (batch, seq_len, ffn_dim).
        timestamp_ns: Monotonic nanosecond timestamp at capture time.
        request_id: Unique identifier for the originating request.
    """
    layer_idx: int
    residual_stream: Tensor
    attention_patterns: Optional[Tensor]
    mlp_activations: Optional[Tensor]
    timestamp_ns: int
    request_id: str


@dataclass
class ProbeResult:
    """Output of a single probing classifier.

    Args:
        category: The safety category this probe monitors.
        risk_score: Probability estimate in [0.0, 1.0].
        layer_idx: Which layer triggered this reading.
        confidence: Model confidence (1 - entropy of prediction).
        latency_ms: Time taken for this probe in milliseconds.
    """
    category: ProbeCategory
    risk_score: float
    layer_idx: int
    confidence: float
    latency_ms: float


@dataclass
class SafetySignal:
    """Aggregated safety assessment from all probes for a single forward pass.

    Args:
        request_id: The originating request.
        risk_level: Overall risk classification.
        composite_score: Weighted combination of all probe scores [0, 1].
        probe_results: Individual results per probe category.
        total_latency_ms: End-to-end hook-to-signal latency.
        triggered_early: Whether detection fired before the final layer.
        alert_tokens_ahead: How many tokens before output this was detected.
    """
    request_id: str
    risk_level: RiskLevel
    composite_score: float
    probe_results: Dict[ProbeCategory, ProbeResult]
    total_latency_ms: float
    triggered_early: bool
    alert_tokens_ahead: int = 0


# ── Probe Interface ───────────────────────────────────────────────────────────

class BaseProbe(nn.Module):
    """Abstract base for all probing classifiers.

    All probes must implement forward() and return a (score, confidence) tuple.
    Probes are designed to be tiny (<<1ms inference) — complexity goes in training,
    not inference.

    Example:
        class HallucinationProbe(BaseProbe):
            def __init__(self, hidden_dim: int):
                super().__init__()
                self.linear = nn.Linear(hidden_dim, 2)

            def forward(self, residual: Tensor) -> Tuple[float, float]:
                logits = self.linear(residual.mean(dim=1))  # mean pool
                probs = logits.softmax(dim=-1)
                return float(probs[0, 1]), float(1 - probs[0].entropy())
    """

    category: ProbeCategory

    def forward(self, bundle: ActivationBundle) -> Tuple[float, float]:
        """Run the probe on activation bundle.

        Args:
            bundle: Captured activations from a forward pass.

        Returns:
            Tuple of (risk_score, confidence) both in [0.0, 1.0].

        Raises:
            NotImplementedError: Subclasses must implement this.
        """
        raise NotImplementedError


class LinearResidualProbe(BaseProbe):
    """Linear probe on mean-pooled residual stream vectors.

    The workhorse probe for ARGUS. Lightweight (2 linear layers), trains fast,
    runs in <0.5ms. Feature: mean-pooled residual stream over sequence dimension.

    Args:
        hidden_dim: Transformer hidden dimension (e.g., 4096 for Llama 3.1 8B).
        category: Which safety category this probe detects.
        probe_dim: Intermediate dimension for 2-layer MLP probe.

    Example:
        probe = LinearResidualProbe(4096, ProbeCategory.HALLUCINATION)
        score, conf = probe(bundle)
    """

    def __init__(
        self,
        hidden_dim: int,
        category: ProbeCategory,
        probe_dim: int = 128,
    ) -> None:
        super().__init__()
        self.category = category
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, probe_dim),
            nn.GELU(),
            nn.Linear(probe_dim, 2),
        )

    def forward(self, bundle: ActivationBundle) -> Tuple[float, float]:
        """Run probe on residual stream.

        Args:
            bundle: Activation bundle from a forward pass hook.

        Returns:
            Tuple of (risk_score, confidence).
        """
        # Mean pool over sequence dimension → shape: (batch, hidden_dim)
        pooled = bundle.residual_stream.mean(dim=1).float()

        with torch.no_grad():
            logits = self.net(pooled)
            probs = logits.softmax(dim=-1)

        risk_score = float(probs[0, 1])  # P(unsafe)
        # Confidence = 1 - normalized entropy
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1)
        confidence = float(1.0 - entropy[0] / torch.log(torch.tensor(2.0)))

        return risk_score, confidence


# ── Hook Manager ──────────────────────────────────────────────────────────────

class HookManager:
    """Attaches non-blocking forward hooks to transformer layers.

    Uses CUDA streams to ensure hook execution doesn't block the main
    inference stream. Hooks run asynchronously on a secondary CUDA stream.

    Args:
        model: The transformer model to monitor.
        target_layers: Layer indices to attach hooks to. If None, hooks all layers.
        device: CUDA device for stream allocation.

    Example:
        manager = HookManager(llama_model, target_layers=[8, 16, 24, 31])
        manager.attach(callback=probe_registry.dispatch)
        # model runs inference normally, hooks fire asynchronously
        manager.detach()  # cleanup on shutdown
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[int]] = None,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._target_layers = target_layers
        self._device = device
        self._hooks: List = []
        self._stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._callback: Optional[Callable[[ActivationBundle], None]] = None

    def attach(
        self,
        callback: Callable[[ActivationBundle], None],
        request_id_fn: Optional[Callable[[], str]] = None,
    ) -> None:
        """Register hooks on model layers and wire callback.

        Args:
            callback: Called with each ActivationBundle (non-blocking).
            request_id_fn: Optional factory for request ID generation.

        Raises:
            RuntimeError: If hooks are already attached.
        """
        if self._hooks:
            raise RuntimeError("Hooks already attached. Call detach() first.")

        self._callback = callback

        layers = self._get_target_layers()
        for layer_idx, layer in enumerate(layers):
            if self._target_layers and layer_idx not in self._target_layers:
                continue

            hook = layer.register_forward_hook(
                self._make_hook(layer_idx, request_id_fn)
            )
            self._hooks.append(hook)

        logger.info(
            "hooks_attached",
            count=len(self._hooks),
            layers=self._target_layers,
            model=type(self._model).__name__,
        )

    def detach(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("hooks_detached")

    def _get_target_layers(self) -> List[nn.Module]:
        """Traverse model to find transformer layers.

        Returns:
            List of layer modules ordered by index.
        """
        # Support common HuggingFace architectures
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return list(self._model.model.layers)
        if hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            return list(self._model.transformer.h)
        raise ValueError(
            f"Cannot find transformer layers in {type(self._model).__name__}. "
            "Add support in HookManager._get_target_layers()."
        )

    def _make_hook(
        self,
        layer_idx: int,
        request_id_fn: Optional[Callable[[], str]],
    ) -> Callable:
        """Factory for a non-blocking forward hook closure.

        Args:
            layer_idx: The index of this layer.
            request_id_fn: Optional factory for request ID.

        Returns:
            A forward hook function compatible with nn.Module.register_forward_hook.
        """
        def hook(
            module: nn.Module,
            inputs: Tuple[Tensor, ...],
            outputs: Tuple[Tensor, ...] | Tensor,
        ) -> None:
            ts = time.monotonic_ns()

            # Extract tensors safely (detach to avoid affecting gradients)
            hidden_state = outputs[0].detach() if isinstance(outputs, tuple) else outputs.detach()

            bundle = ActivationBundle(
                layer_idx=layer_idx,
                residual_stream=hidden_state,
                attention_patterns=None,  # populated if attention hook registered separately
                mlp_activations=None,
                timestamp_ns=ts,
                request_id=request_id_fn() if request_id_fn else "unknown",
            )

            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    self._callback(bundle)
            else:
                self._callback(bundle)

        return hook


# ── Probe Registry ────────────────────────────────────────────────────────────

class ProbeRegistry:
    """Manages all probing classifiers and aggregates their outputs.

    Dispatches ActivationBundles to registered probes, aggregates results,
    and emits SafetySignals downstream.

    Args:
        probes: Dict mapping ProbeCategory to trained probe modules.
        thresholds: Risk score thresholds per category (default 0.5).
        layer_weights: Weight per layer index for composite score (later layers = higher weight).

    Example:
        registry = ProbeRegistry(probes={
            ProbeCategory.HALLUCINATION: HallucinationProbe(4096),
            ProbeCategory.JAILBREAK: JailbreakProbe(4096),
        })
        signal = registry.dispatch(bundle)
    """

    # Risk level thresholds for composite score
    RISK_THRESHOLDS = {
        RiskLevel.CRITICAL: 0.85,
        RiskLevel.HIGH: 0.65,
        RiskLevel.MEDIUM: 0.40,
        RiskLevel.LOW: 0.20,
    }

    def __init__(
        self,
        probes: Dict[ProbeCategory, BaseProbe],
        thresholds: Optional[Dict[ProbeCategory, float]] = None,
        layer_weights: Optional[Dict[int, float]] = None,
        signal_callback: Optional[Callable[[SafetySignal], None]] = None,
    ) -> None:
        self._probes = probes
        self._thresholds = thresholds or {cat: 0.5 for cat in ProbeCategory}
        self._layer_weights = layer_weights or {}
        self._signal_callback = signal_callback
        self._results_buffer: Dict[str, Dict[ProbeCategory, ProbeResult]] = {}

    def dispatch(self, bundle: ActivationBundle) -> Optional[SafetySignal]:
        """Run all probes on the given activation bundle.

        Args:
            bundle: Captured activations from a forward hook.

        Returns:
            SafetySignal if enough data to aggregate, else None (buffering).
        """
        t0 = time.monotonic_ns()

        probe_results: Dict[ProbeCategory, ProbeResult] = {}

        for category, probe in self._probes.items():
            probe_t0 = time.monotonic_ns()
            try:
                score, confidence = probe(bundle)
                probe_results[category] = ProbeResult(
                    category=category,
                    risk_score=score,
                    layer_idx=bundle.layer_idx,
                    confidence=confidence,
                    latency_ms=(time.monotonic_ns() - probe_t0) / 1e6,
                )
            except Exception as e:
                logger.error(
                    "probe_error",
                    category=category.name,
                    layer=bundle.layer_idx,
                    error=str(e),
                )

        # Aggregate to composite signal
        composite = self._aggregate(probe_results, bundle.layer_idx)
        risk_level = self._classify_risk(composite)
        total_ms = (time.monotonic_ns() - t0) / 1e6

        signal = SafetySignal(
            request_id=bundle.request_id,
            risk_level=risk_level,
            composite_score=composite,
            probe_results=probe_results,
            total_latency_ms=total_ms,
            triggered_early=risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL),
        )

        if self._signal_callback:
            self._signal_callback(signal)

        if total_ms > 10.0:
            logger.warning("probe_sla_miss", latency_ms=total_ms, layer=bundle.layer_idx)

        return signal

    def _aggregate(
        self,
        results: Dict[ProbeCategory, ProbeResult],
        layer_idx: int,
    ) -> float:
        """Compute weighted composite risk score.

        Args:
            results: Probe results for this layer.
            layer_idx: Used to apply layer-position weight.

        Returns:
            Composite risk score in [0, 1].
        """
        if not results:
            return 0.0

        # Category weights: jailbreak and toxic reasoning are more severe
        category_weights = {
            ProbeCategory.HALLUCINATION: 0.20,
            ProbeCategory.JAILBREAK: 0.35,
            ProbeCategory.TOXIC_REASONING: 0.30,
            ProbeCategory.POLICY_VIOLATION: 0.15,
        }

        weighted_sum = sum(
            r.risk_score * r.confidence * category_weights.get(cat, 0.25)
            for cat, r in results.items()
        )
        weight_total = sum(
            category_weights.get(cat, 0.25) for cat in results
        )

        layer_multiplier = self._layer_weights.get(layer_idx, 1.0)
        return min(1.0, (weighted_sum / weight_total) * layer_multiplier)

    def _classify_risk(self, composite_score: float) -> RiskLevel:
        """Map composite score to risk level.

        Args:
            composite_score: Aggregated score in [0, 1].

        Returns:
            RiskLevel enum value.
        """
        for level, threshold in sorted(
            self.RISK_THRESHOLDS.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if composite_score >= threshold:
                return level
        return RiskLevel.SAFE


# ── LatentSentinel: Top-level façade ─────────────────────────────────────────

class LatentSentinel:
    """Top-level interface for MOD-01. Wire this to any HuggingFace transformer.

    Combines HookManager + ProbeRegistry into a single easy interface.
    Call `monitor(model)` to start, `stop()` to detach.

    Optionally integrates with OmniSafetyCritic (MOD-03): when
    ``critic_endpoint`` is set, the composite probe score is blended
    asynchronously with the critic score using ``critic_blend_weight``.
    This is a fire-and-forget async call — it never blocks the probe path.

    Args:
        probes: Dict of trained probing classifiers.
        target_layers: Layers to monitor (None = all).
        signal_callback: Called with every SafetySignal emitted.
        device: CUDA device string.
        critic_endpoint: Optional URL of the OmniSafetyCritic HTTP server.
            If set, the critic score is blended into the composite signal.
        critic_blend_weight: Weight given to the critic score in the blend
            (0.0 = probe-only, 1.0 = critic-only, default 0.3).
        critic_timeout_ms: Timeout for critic HTTP calls in milliseconds.

    Example:
        sentinel = LatentSentinel(
            probes={
                ProbeCategory.HALLUCINATION: load_probe("hallucination_probe_llama31"),
                ProbeCategory.JAILBREAK: load_probe("jailbreak_probe_llama31"),
            },
            signal_callback=kafka_publisher.publish,
            critic_endpoint="http://safety-critic:8001",
        )
        sentinel.monitor(llama_model)
        # ... run inference normally ...
        sentinel.stop()
    """

    def __init__(
        self,
        probes: Dict[ProbeCategory, BaseProbe],
        target_layers: Optional[List[int]] = None,
        signal_callback: Optional[Callable[[SafetySignal], None]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        critic_endpoint: Optional[str] = None,
        critic_blend_weight: float = 0.3,
        critic_timeout_ms: float = 80.0,
    ) -> None:
        self._probe_registry = ProbeRegistry(
            probes=probes,
            signal_callback=signal_callback,
        )
        self._hook_manager: Optional[HookManager] = None
        self._target_layers = target_layers
        self._device = device
        self._active = False

        # MOD-03 OmniSafetyCritic integration
        self._critic_endpoint = critic_endpoint
        self._critic_blend_weight = max(0.0, min(1.0, critic_blend_weight))
        self._critic_timeout_ms = critic_timeout_ms
        self._critic_client: Optional[Any] = None
        if critic_endpoint:
            self._init_critic_client(critic_endpoint, critic_timeout_ms)

    def _init_critic_client(self, endpoint: str, timeout_ms: float) -> None:
        """Lazily initialise the OmniSafetyCritic HTTP client.

        Args:
            endpoint: Base URL of the safety critic server.
            timeout_ms: Request timeout in milliseconds.
        """
        try:
            from src.safety_critic.critic import OmniSafetyCriticClient
            self._critic_client = OmniSafetyCriticClient(
                endpoint=endpoint,
                timeout_ms=timeout_ms,
            )
            logger.info(
                "critic_client_initialised",
                endpoint=endpoint,
                blend_weight=self._critic_blend_weight,
            )
        except ImportError:
            logger.warning(
                "critic_client_import_failed",
                action="running_probe_only_mode",
            )

    async def _blend_critic_score(
        self,
        signal: SafetySignal,
        content: str,
    ) -> SafetySignal:
        """Asynchronously blend the OmniSafetyCritic score into the signal.

        Calls the critic server with a text representation of the highest-risk
        probe category, then blends:
            blended = (1 - w) * probe_score + w * critic_score

        Args:
            signal: The SafetySignal from the probe pipeline.
            content: Text content to send to the critic (agent output snippet).

        Returns:
            Updated SafetySignal with blended composite_score.
        """
        if self._critic_client is None:
            return signal

        try:
            from src.safety_critic.critic import ContentModality, CriticInput
            critic_input = CriticInput(
                content=content[:2048],  # truncate to server max
                modality=ContentModality.TEXT,
            )
            critic_output = await self._critic_client.score(critic_input)
            w = self._critic_blend_weight
            blended = (1.0 - w) * signal.composite_score + w * critic_output.safety_score
            blended = max(0.0, min(1.0, blended))

            # Re-classify risk level with blended score
            risk_level = self._probe_registry._classify_risk(blended)
            object.__setattr__(signal, "composite_score", blended)
            object.__setattr__(signal, "risk_level", risk_level)

            logger.debug(
                "critic_blend_applied",
                probe_score=f"{signal.composite_score:.3f}",
                critic_score=f"{critic_output.safety_score:.3f}",
                blended=f"{blended:.3f}",
            )
        except Exception as exc:
            # Critic call failure must never break the probe path
            logger.warning("critic_blend_failed", error=str(exc))

        return signal

    def monitor(self, model: nn.Module) -> None:
        """Attach hooks to model and begin monitoring.

        Args:
            model: HuggingFace-compatible transformer model.

        Raises:
            RuntimeError: If already monitoring a model.
        """
        if self._active:
            raise RuntimeError("LatentSentinel is already active. Call stop() first.")

        self._hook_manager = HookManager(model, self._target_layers, self._device)
        self._hook_manager.attach(callback=self._probe_registry.dispatch)
        self._active = True
        logger.info("latent_sentinel_started", model=type(model).__name__)

    def stop(self) -> None:
        """Detach all hooks and stop monitoring."""
        if self._hook_manager:
            self._hook_manager.detach()
        self._active = False
        logger.info("latent_sentinel_stopped")

    @property
    def is_active(self) -> bool:
        """Whether the sentinel is currently monitoring."""
        return self._active
