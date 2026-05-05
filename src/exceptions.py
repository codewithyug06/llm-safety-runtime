"""
ARGUS Custom Exception Hierarchy
=================================
All production errors must raise from this module.
Handlers emit structured logs + increment Prometheus counters.
"""

from __future__ import annotations


# ── Base ──────────────────────────────────────────────────────────────────────

class ArgusBaseException(Exception):
    """Root exception for all ARGUS errors.

    Args:
        message: Human-readable error description.
        module: ARGUS module that raised the error (e.g. "latent_sentinel").
    """

    def __init__(self, message: str, module: str = "argus") -> None:
        super().__init__(message)
        self.message = message
        self.module = module

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module={self.module!r}, message={self.message!r})"


# ── MOD-01: LatentSentinel ────────────────────────────────────────────────────

class LatentSentinelError(ArgusBaseException):
    """Base for all LatentSentinel errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="latent_sentinel")


class HookRegistrationError(LatentSentinelError):
    """Raised when a forward hook cannot be attached to a model layer.

    Example:
        >>> raise HookRegistrationError("Layer 16 not found in model")
    """


class HookAlreadyRegisteredError(LatentSentinelError):
    """Raised when attempting to register hooks on an already-hooked model."""


class ProbeInferenceError(LatentSentinelError):
    """Raised when a probing classifier fails during forward pass.

    Args:
        category: Probe category that failed (e.g. "hallucination").
    """

    def __init__(self, message: str, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


class ProbeNotLoadedError(LatentSentinelError):
    """Raised when a probe model file is missing or corrupt."""


class SLAViolationError(LatentSentinelError):
    """Raised when hook-to-signal latency exceeds the p99 budget (15ms).

    Args:
        latency_ms: Measured latency in milliseconds.
        budget_ms: Configured budget in milliseconds.
    """

    def __init__(self, latency_ms: float, budget_ms: float) -> None:
        super().__init__(
            f"SLA violation: {latency_ms:.2f}ms > {budget_ms:.2f}ms budget"
        )
        self.latency_ms = latency_ms
        self.budget_ms = budget_ms


# ── MOD-02: CausalInterventionEngine ─────────────────────────────────────────

class CausalEngineError(ArgusBaseException):
    """Base for all CausalInterventionEngine errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="causal_engine")


class CausalScrubError(CausalEngineError):
    """Raised when the causal scrubbing / ablation study fails."""


class InterventionError(CausalEngineError):
    """Raised when a soft weight intervention cannot be applied.

    Args:
        head_layer: Transformer layer index of the target head.
        head_idx: Attention head index within the layer.
    """

    def __init__(self, message: str, head_layer: int = -1, head_idx: int = -1) -> None:
        super().__init__(message)
        self.head_layer = head_layer
        self.head_idx = head_idx


class CausalGraphNotFoundError(CausalEngineError):
    """Raised when no causal graph exists for the given model checkpoint."""


# ── MOD-03: OmniSafetyCritic ─────────────────────────────────────────────────

class SafetyCriticError(ArgusBaseException):
    """Base for all OmniSafetyCritic errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="safety_critic")


class CriticServingError(SafetyCriticError):
    """Raised when the vLLM serving endpoint is unavailable or returns an error."""


class CriticInferenceTimeoutError(SafetyCriticError):
    """Raised when safety critic inference exceeds the SLA budget (80ms p95).

    Args:
        latency_ms: Measured inference latency.
    """

    def __init__(self, latency_ms: float) -> None:
        super().__init__(f"Critic inference timeout: {latency_ms:.1f}ms > 80ms SLA")
        self.latency_ms = latency_ms


class DatasetCurationError(SafetyCriticError):
    """Raised during DPO dataset construction if source data is malformed."""


# ── MOD-04: FederatedRLHF ────────────────────────────────────────────────────

class FederatedRLHFError(ArgusBaseException):
    """Base for all FederatedRLHF errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="federated_rlhf")


class FederatedRoundError(FederatedRLHFError):
    """Raised when a federated training round fails or cannot reach quorum.

    Args:
        round_num: The federated round number that failed.
        available_clients: Number of clients that responded.
        required_clients: Minimum clients required.
    """

    def __init__(
        self,
        message: str,
        round_num: int = -1,
        available_clients: int = 0,
        required_clients: int = 0,
    ) -> None:
        super().__init__(message)
        self.round_num = round_num
        self.available_clients = available_clients
        self.required_clients = required_clients


class PrivacyBudgetExhaustedError(FederatedRLHFError):
    """Raised when cumulative DP-SGD epsilon exceeds the configured budget.

    Args:
        current_epsilon: Accumulated privacy cost so far.
        budget_epsilon: Maximum allowed epsilon (default 3.0).
    """

    def __init__(self, current_epsilon: float, budget_epsilon: float = 3.0) -> None:
        super().__init__(
            f"Privacy budget exhausted: ε={current_epsilon:.4f} > limit={budget_epsilon}"
        )
        self.current_epsilon = current_epsilon
        self.budget_epsilon = budget_epsilon


class GradientAggregationError(FederatedRLHFError):
    """Raised when federated gradient aggregation produces NaN/Inf tensors."""


# ── MOD-05: PredictiveOracle ─────────────────────────────────────────────────

class PredictiveOracleError(ArgusBaseException):
    """Base for all PredictiveOracle errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="predictive_oracle")


class OraclePredictionError(PredictiveOracleError):
    """Raised when the oracle model fails to produce a valid prediction."""


class InsufficientTelemetryError(PredictiveOracleError):
    """Raised when the telemetry window has too few timesteps for inference.

    Args:
        available: Number of timesteps available.
        required: Minimum timesteps required.
    """

    def __init__(self, available: int, required: int) -> None:
        super().__init__(
            f"Insufficient telemetry: {available} timesteps < {required} required"
        )
        self.available = available
        self.required = required


class ConformalCalibrationError(PredictiveOracleError):
    """Raised when conformal calibration set is too small or contains invalid data."""


# ── MOD-06: AutonomousRemediator ─────────────────────────────────────────────

class RemediatorError(ArgusBaseException):
    """Base for all AutonomousRemediator errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="autonomous_remediator")


class RemediationError(RemediatorError):
    """Raised when a remediation action fails to execute.

    Args:
        action: The remediation action that failed.
        agent_id: ID of the misbehaving agent.
    """

    def __init__(self, message: str, action: str = "unknown", agent_id: str = "") -> None:
        super().__init__(message)
        self.action = action
        self.agent_id = agent_id


class AuditLogError(RemediatorError):
    """Raised when writing to Cloud Spanner audit log fails."""


class QuarantineError(RemediatorError):
    """Raised when agent quarantine via Redis fails."""


class RollbackError(RemediatorError):
    """Raised when model rollback via MLflow fails.

    Args:
        model_name: Name of the model that failed to roll back.
        target_version: The version we tried to restore.
    """

    def __init__(self, message: str, model_name: str = "", target_version: str = "") -> None:
        super().__init__(message)
        self.model_name = model_name
        self.target_version = target_version


class EscalationError(RemediatorError):
    """Raised when human escalation notification (PagerDuty/Slack) fails."""


# ── Infrastructure / Config ───────────────────────────────────────────────────

class ConfigurationError(ArgusBaseException):
    """Raised when a required configuration value is missing or invalid.

    Args:
        config_key: The YAML key or env var that is invalid.
    """

    def __init__(self, message: str, config_key: str = "") -> None:
        super().__init__(message, module="config")
        self.config_key = config_key


class KafkaConnectionError(ArgusBaseException):
    """Raised when the Kafka producer/consumer cannot connect to the broker."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="infra")


class RedisConnectionError(ArgusBaseException):
    """Raised when the Redis client cannot connect."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="infra")


class MLflowRegistryError(ArgusBaseException):
    """Raised when MLflow model registry operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, module="mlops")
