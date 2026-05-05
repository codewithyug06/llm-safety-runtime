"""
MLOps: ARGUS Model Registry Wrapper
=====================================
Wraps mlflow.MlflowClient to provide a clean interface for:
  - Registering trained model artifacts
  - Promoting models through staging → production
  - Gating promotion on safety metric thresholds
  - Rolling back to previous versions (called by RollbackAgent)
  - Listing all versions and their metrics

All promotion decisions are logged with structured metadata for audit.

Example:
    registry = ArgusModelRegistry("http://mlflow:5000")
    registry.register_model(run_id="abc123", model_name="argus-safety-critic")
    registry.promote_to_production("argus-safety-critic", min_safety_accuracy=0.85)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from src.exceptions import MLflowRegistryError

logger = structlog.get_logger(__name__)

# Safety gate: minimum accuracy required to promote to Production
DEFAULT_MIN_SAFETY_ACCURACY: float = 0.85
DEFAULT_MIN_F1: float = 0.78


@dataclass
class ModelVersion:
    """A registered model version in the MLflow registry.

    Args:
        model_name: Registry model name.
        version: Version string.
        stage: Current lifecycle stage.
        run_id: MLflow run that produced this version.
        metrics: Dict of evaluation metrics.
        creation_time: ISO-8601 creation timestamp.
    """

    model_name: str
    version: str
    stage: str
    run_id: str
    metrics: Dict[str, float]
    creation_time: str


class ArgusModelRegistry:
    """Production-grade MLflow model registry for ARGUS safety models.

    Wraps mlflow.MlflowClient with:
    - Safety metric gating before promotion
    - Structured audit logging on every state transition
    - Multi-model support (safety critic, oracle, probes)

    Args:
        tracking_uri: MLflow tracking server URI.
        min_safety_accuracy: Minimum accuracy to allow Production promotion.
        min_f1: Minimum F1 score to allow Production promotion.

    Example:
        registry = ArgusModelRegistry("http://mlflow:5000")
        registry.register_model("abc123", "argus-safety-critic",
                                artifact_path="safety_critic")
        registry.promote_to_production("argus-safety-critic")
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        min_safety_accuracy: float = DEFAULT_MIN_SAFETY_ACCURACY,
        min_f1: float = DEFAULT_MIN_F1,
    ) -> None:
        self._tracking_uri = tracking_uri
        self.min_safety_accuracy = min_safety_accuracy
        self.min_f1 = min_f1
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazily create MLflow client."""
        if self._client is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
                self._client = mlflow.MlflowClient()
            except ImportError:
                raise ImportError("Run: pip install mlflow>=2.10.0")
        return self._client

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a model artifact from an MLflow run.

        Args:
            run_id: MLflow run ID containing the artifact.
            model_name: Name to register under in the registry.
            artifact_path: Path within the run's artifact store.
            tags: Optional key-value tags for this version.

        Returns:
            Registered model version string (e.g. "3").

        Raises:
            MLflowRegistryError: If registration fails.
        """
        try:
            import mlflow
            model_uri = f"runs:/{run_id}/{artifact_path}"
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            version = mv.version

            client = self._get_client()
            # Set to Staging first
            client.transition_model_version_stage(
                name=model_name, version=version, stage="Staging"
            )
            if tags:
                for k, v in tags.items():
                    client.set_model_version_tag(model_name, version, k, v)

            logger.info(
                "model_registered",
                model=model_name,
                version=version,
                run_id=run_id,
                stage="Staging",
            )
            return version

        except Exception as exc:
            raise MLflowRegistryError(
                f"Failed to register {model_name} from run {run_id}: {exc}"
            ) from exc

    def promote_to_production(
        self,
        model_name: str,
        version: Optional[str] = None,
        min_safety_accuracy: Optional[float] = None,
        min_f1: Optional[float] = None,
        force: bool = False,
    ) -> str:
        """Promote a model version to Production (with safety metric gate).

        If version is None, promotes the latest Staging version.
        Safety gate: requires safety_accuracy ≥ min_safety_accuracy and f1 ≥ min_f1.

        Args:
            model_name: Registered model name.
            version: Version to promote (None = latest Staging).
            min_safety_accuracy: Minimum accuracy threshold.
            min_f1: Minimum F1 threshold.
            force: Skip safety metric gate if True (use carefully).

        Returns:
            Promoted version string.

        Raises:
            MLflowRegistryError: If gate fails or promotion errors.
        """
        acc_threshold = min_safety_accuracy or self.min_safety_accuracy
        f1_threshold = min_f1 or self.min_f1

        try:
            client = self._get_client()

            # Find version to promote
            if version is None:
                staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
                if not staging_versions:
                    raise MLflowRegistryError(f"No Staging version found for {model_name}")
                version = staging_versions[0].version

            # Check metrics gate
            if not force:
                metrics = self._get_run_metrics(client, model_name, version)
                accuracy = metrics.get("safety_accuracy", metrics.get("eval_accuracy", 0.0))
                f1 = metrics.get("f1", metrics.get("eval_f1", 0.0))

                if accuracy < acc_threshold:
                    raise MLflowRegistryError(
                        f"Promotion blocked: safety_accuracy={accuracy:.4f} < {acc_threshold}"
                    )
                if f1 < f1_threshold:
                    raise MLflowRegistryError(
                        f"Promotion blocked: f1={f1:.4f} < {f1_threshold}"
                    )

                logger.info(
                    "safety_gate_passed",
                    model=model_name,
                    version=version,
                    accuracy=f"{accuracy:.4f}",
                    f1=f"{f1:.4f}",
                )

            # Archive current Production
            current_prod = client.get_latest_versions(model_name, stages=["Production"])
            for cv in current_prod:
                client.transition_model_version_stage(
                    name=model_name, version=cv.version, stage="Archived"
                )
                logger.info("previous_version_archived", model=model_name, version=cv.version)

            # Promote to Production
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )

            logger.info(
                "model_promoted_to_production",
                model=model_name,
                version=version,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            return version

        except MLflowRegistryError:
            raise
        except Exception as exc:
            raise MLflowRegistryError(
                f"Promotion failed for {model_name} v{version}: {exc}"
            ) from exc

    def rollback_to_version(
        self,
        model_name: str,
        version: int,
    ) -> None:
        """Roll back a model to a specific version (called by RollbackAgent).

        Args:
            model_name: Registered model name.
            version: Target version number.

        Raises:
            MLflowRegistryError: If rollback fails.
        """
        try:
            client = self._get_client()

            # Archive current Production
            current = client.get_latest_versions(model_name, stages=["Production"])
            for cv in current:
                client.transition_model_version_stage(
                    name=model_name, version=cv.version, stage="Archived"
                )

            # Restore target version
            client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage="Production",
                archive_existing_versions=True,
            )

            logger.warning(
                "model_rolled_back",
                model=model_name,
                target_version=version,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as exc:
            raise MLflowRegistryError(
                f"Rollback to v{version} failed for {model_name}: {exc}"
            ) from exc

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a registered model.

        Args:
            model_name: Registered model name.

        Returns:
            List of ModelVersion dataclasses.
        """
        try:
            client = self._get_client()
            versions = client.search_model_versions(f"name='{model_name}'")
            result = []
            for v in versions:
                metrics = self._get_run_metrics(client, model_name, v.version)
                result.append(ModelVersion(
                    model_name=model_name,
                    version=v.version,
                    stage=v.current_stage,
                    run_id=v.run_id,
                    metrics=metrics,
                    creation_time=str(v.creation_timestamp),
                ))
            return sorted(result, key=lambda x: int(x.version), reverse=True)
        except Exception as exc:
            logger.error("list_versions_failed", model=model_name, error=str(exc))
            return []

    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently Production version.

        Args:
            model_name: Registered model name.

        Returns:
            ModelVersion or None if no Production version.
        """
        try:
            client = self._get_client()
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                return None
            v = versions[0]
            metrics = self._get_run_metrics(client, model_name, v.version)
            return ModelVersion(
                model_name=model_name,
                version=v.version,
                stage="Production",
                run_id=v.run_id,
                metrics=metrics,
                creation_time=str(v.creation_timestamp),
            )
        except Exception as exc:
            logger.error("get_production_failed", model=model_name, error=str(exc))
            return None

    def _get_run_metrics(
        self, client: Any, model_name: str, version: str
    ) -> Dict[str, float]:
        """Fetch metrics from the MLflow run linked to a model version.

        Args:
            client: MLflow client.
            model_name: Model name.
            version: Version string.

        Returns:
            Dict of metric name → float value.
        """
        try:
            mv = client.get_model_version(model_name, version)
            run = client.get_run(mv.run_id)
            return {k: float(v) for k, v in run.data.metrics.items()}
        except Exception:
            return {}
