"""
ARGUS Configuration Loader
============================
Loads YAML config files from configs/ and exposes them as Pydantic Settings models.
All modules import their config from here — never hardcode values.

Usage:
    from src.config import load_sentinel_config, load_oracle_config
    cfg = load_sentinel_config()
    threshold = cfg.probes.hallucination.threshold
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

import structlog

logger = structlog.get_logger(__name__)

# Root of the project (one level above src/)
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, expanding environment variables in string values.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed YAML as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        raw = f.read()
    # Expand ${ENV_VAR} placeholders
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)


# ── MOD-01: LatentSentinel ────────────────────────────────────────────────────

class ProbeConfig(BaseModel):
    enabled: bool = True
    model_path: str = ""
    probe_dim: int = 128
    threshold: float = 0.5

    @field_validator("threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v


class ProbesConfig(BaseModel):
    hallucination: ProbeConfig = Field(default_factory=ProbeConfig)
    jailbreak: ProbeConfig = Field(default_factory=ProbeConfig)
    toxic_reasoning: ProbeConfig = Field(default_factory=ProbeConfig)
    policy_violation: ProbeConfig = Field(default_factory=ProbeConfig)


class HooksConfig(BaseModel):
    target_layers: List[int] = Field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 28, 31])
    use_cuda_streams: bool = True


class AggregationConfig(BaseModel):
    category_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "hallucination": 0.20,
            "jailbreak": 0.35,
            "toxic_reasoning": 0.30,
            "policy_violation": 0.15,
        }
    )
    layer_weights: Dict[str, float] = Field(
        default_factory=lambda: {"31": 1.5, "28": 1.3, "24": 1.1, "default": 1.0}
    )


class SLAConfig(BaseModel):
    max_latency_ms: float = 10.0
    warn_latency_ms: float = 8.0


class KafkaConfig(BaseModel):
    topic: str = "argus.safety.signals"
    bootstrap_servers: str = "localhost:9092"
    producer_config: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"


class LatentSentinelConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    probes: ProbesConfig = Field(default_factory=ProbesConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    sla: SLAConfig = Field(default_factory=SLAConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)


def load_sentinel_config() -> LatentSentinelConfig:
    """Load and validate the LatentSentinel configuration.

    Returns:
        Validated LatentSentinelConfig instance.
    """
    data = _load_yaml(_CONFIGS_DIR / "latent_sentinel.yaml")
    cfg = LatentSentinelConfig(**data)
    logger.info("latent_sentinel_config_loaded", layers=cfg.hooks.target_layers)
    return cfg


# ── MOD-03: OmniSafetyCritic ─────────────────────────────────────────────────

class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"


class DPOConfig(BaseModel):
    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024
    loss_type: str = "sigmoid"


class TrainingConfig(BaseModel):
    output_dir: str = "models/safety_critic"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


class SafetyCriticConfig(BaseModel):
    base_model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow_experiment: str = "argus/safety_critic/dpo"
    wandb_project: str = "argus-safety-critic"
    serving_port: int = 8001
    serving_max_model_len: int = 2048
    serving_gpu_memory_utilization: float = 0.85


def load_safety_critic_config() -> SafetyCriticConfig:
    """Load and validate the OmniSafetyCritic configuration.

    Returns:
        Validated SafetyCriticConfig instance.
    """
    data = _load_yaml(_CONFIGS_DIR / "safety_critic.yaml")
    cfg = SafetyCriticConfig(**data)
    logger.info("safety_critic_config_loaded", base_model=cfg.base_model.name)
    return cfg


# ── MOD-04: FederatedRLHF ────────────────────────────────────────────────────

class FederatedCoordinatorConfig(BaseModel):
    strategy: str = "fedavg"
    num_rounds: int = 50
    min_clients: int = 2
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    server_address: str = "0.0.0.0:8080"


class DifferentialPrivacyConfig(BaseModel):
    enabled: bool = True
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    target_epsilon: float = 3.0
    target_delta: float = 1e-5


class LocalTrainingConfig(BaseModel):
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 3e-5


class FederatedRLHFConfig(BaseModel):
    coordinator: FederatedCoordinatorConfig = Field(default_factory=FederatedCoordinatorConfig)
    differential_privacy: DifferentialPrivacyConfig = Field(
        default_factory=DifferentialPrivacyConfig
    )
    local_training: LocalTrainingConfig = Field(default_factory=LocalTrainingConfig)
    mlflow_experiment: str = "argus/federated_rlhf"
    kubeflow_pipeline_name: str = "argus-federated-rlhf-v1"
    kubeflow_namespace: str = "argus-prod"
    promotion_safety_accuracy_threshold: float = 0.85
    promotion_fnr_threshold: float = 0.10


def load_federated_rlhf_config() -> FederatedRLHFConfig:
    """Load and validate the FederatedRLHF configuration.

    Returns:
        Validated FederatedRLHFConfig instance.
    """
    data = _load_yaml(_CONFIGS_DIR / "federated_rlhf.yaml")
    cfg = FederatedRLHFConfig(**data)
    logger.info(
        "federated_rlhf_config_loaded",
        rounds=cfg.coordinator.num_rounds,
        epsilon=cfg.differential_privacy.target_epsilon,
    )
    return cfg


# ── MOD-05: PredictiveOracle ─────────────────────────────────────────────────

class OracleModelConfig(BaseModel):
    hidden_dim: int = 128
    num_heads: int = 8
    num_layers: int = 3
    patch_len: int = 12
    stride: int = 6
    seq_len: int = 60
    num_features: int = 9
    dropout: float = 0.1
    forecast_horizons: List[int] = Field(default_factory=lambda: [30, 60, 90])


class OracleTrainingConfig(BaseModel):
    output_dir: str = "models/oracle"
    num_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    val_split: float = 0.15
    test_split: float = 0.10


class ConformalConfig(BaseModel):
    calibration_alpha: float = 0.1   # 90% coverage
    min_calibration_samples: int = 200


class PredictiveOracleConfig(BaseModel):
    model: OracleModelConfig = Field(default_factory=OracleModelConfig)
    training: OracleTrainingConfig = Field(default_factory=OracleTrainingConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    serving_latency_budget_ms: float = 80.0
    mlflow_experiment: str = "argus/predictive_oracle"
    alert_threshold: float = 0.65
    kafka_telemetry_topic: str = "argus.telemetry"
    kafka_predictions_topic: str = "argus.risk.predictions"


def load_oracle_config() -> PredictiveOracleConfig:
    """Load and validate the PredictiveOracle configuration.

    Returns:
        Validated PredictiveOracleConfig instance.
    """
    data = _load_yaml(_CONFIGS_DIR / "predictive_oracle.yaml")
    cfg = PredictiveOracleConfig(**data)
    logger.info(
        "oracle_config_loaded",
        horizons=cfg.model.forecast_horizons,
        seq_len=cfg.model.seq_len,
    )
    return cfg


# ── MOD-06: AutonomousRemediator ─────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    no_action_below: float = 0.40
    filter_output_below: float = 0.65
    quarantine_below: float = 0.80
    rollback_below: float = 0.90
    # >= rollback_below → escalate_human


class RemediatorServingConfig(BaseModel):
    max_latency_ms: float = 200.0
    llm_model: str = "meta-llama/Llama-3.1-70B-Instruct"
    llm_max_tokens: int = 1024
    spanner_instance: str = ""
    spanner_database: str = ""
    redis_url: str = "redis://localhost:6379"
    slack_webhook_url: str = ""
    pagerduty_routing_key: str = ""


class AutonomousRemediatorConfig(BaseModel):
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    serving: RemediatorServingConfig = Field(default_factory=RemediatorServingConfig)
    kafka_signals_topic: str = "argus.safety.signals"
    kafka_remediation_topic: str = "argus.remediation"
    audit_log_enabled: bool = True
    rate_limit_window_seconds: int = 60
    rate_limit_max_calls: int = 10


def load_remediator_config() -> AutonomousRemediatorConfig:
    """Load and validate the AutonomousRemediator configuration.

    Returns:
        Validated AutonomousRemediatorConfig instance.
    """
    data = _load_yaml(_CONFIGS_DIR / "autonomous_remediator.yaml")
    cfg = AutonomousRemediatorConfig(**data)
    logger.info("remediator_config_loaded", llm_model=cfg.serving.llm_model)
    return cfg


# ── Environment settings (secrets via env vars) ───────────────────────────────

class ArgusEnvSettings(BaseSettings):
    """Environment variable bindings for all secrets and external service URLs.

    Never put these in YAML files — set them in .env or inject via GKE secrets.
    """

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    wandb_api_key: str = Field(default="", alias="WANDB_API_KEY")
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    gke_cluster_name: str = Field(default="", alias="GKE_CLUSTER_NAME")
    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")
    spanner_instance_id: str = Field(default="", alias="SPANNER_INSTANCE_ID")
    spanner_database_id: str = Field(default="", alias="SPANNER_DATABASE_ID")
    hf_token: str = Field(default="", alias="HF_TOKEN")

    model_config = {"env_file": ".env", "populate_by_name": True}


def load_env_settings() -> ArgusEnvSettings:
    """Load environment settings from .env file + real environment variables.

    Returns:
        ArgusEnvSettings with all secret values populated.
    """
    return ArgusEnvSettings()
