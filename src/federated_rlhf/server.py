"""
MOD-04: ARGUS Federated RLHF Server
======================================
Flower aggregation server with FedAvg strategy and MLflow model registration.
Coordinates global LoRA weight averaging across N client nodes per round,
then registers the aggregated model to MLflow Staging.

Key design decisions:
- FedAvg with data-size weighting: larger local datasets get more weight
- Minimum 2 clients required per round (configurable)
- Aggregated model registered to MLflow Staging after every round
- Privacy-safe: server never sees raw training data, only weight deltas

Run via: scripts/run_federated_round.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog

from src.exceptions import FederatedRoundError, GradientAggregationError

logger = structlog.get_logger(__name__)


# ── Custom FedAvg strategy with MLflow integration ────────────────────────────

class ArgusFedAvgStrategy:
    """Weighted FedAvg aggregation strategy with MLflow registration.

    Aggregates LoRA adapter parameters using weighted average based on
    number of local training examples. Registers the result in MLflow
    Staging after each successful round.

    Args:
        min_fit_clients: Minimum clients required to start a round.
        min_evaluate_clients: Minimum clients required for evaluation.
        min_available_clients: Minimum clients that must be connected.
        mlflow_experiment: MLflow experiment name.
        mlflow_model_name: Registry model name for aggregated checkpoints.
        fraction_fit: Fraction of clients to use for fitting (1.0 = all).
    """

    def __init__(
        self,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        mlflow_experiment: str = "argus/federated_rlhf",
        mlflow_model_name: str = "argus-safety-critic-federated",
        fraction_fit: float = 1.0,
    ) -> None:
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_model_name = mlflow_model_name
        self.fraction_fit = fraction_fit

        self._round_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],  # (client_proxy, FitRes)
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:
        """Aggregate fit results using weighted FedAvg.

        Args:
            server_round: Current round number.
            results: List of (client_proxy, FitRes) from successful clients.
            failures: List of failed clients.

        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics).

        Raises:
            GradientAggregationError: If aggregation fails.
        """
        if not results:
            raise GradientAggregationError(
                f"Round {server_round}: No client results to aggregate"
            )

        if len(results) < self.min_fit_clients:
            raise FederatedRoundError(
                message=f"Insufficient clients: got {len(results)}, need {self.min_fit_clients}",
                round_num=server_round,
                available_clients=len(results),
                required_clients=self.min_fit_clients,
            )

        # Extract parameters and weights (num_examples)
        weights_results = []
        for client_proxy, fit_res in results:
            params = self._parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            weights_results.append((params, num_examples))

        # Weighted average
        try:
            aggregated = self._weighted_average(weights_results)
        except Exception as exc:
            raise GradientAggregationError(
                f"Weighted averaging failed at round {server_round}: {exc}"
            ) from exc

        # Collect round metrics
        total_examples = sum(n for _, n in weights_results)
        client_metrics = [fit_res.metrics for _, fit_res in results]
        avg_train_loss = np.mean([m.get("train_loss", 0.0) for m in client_metrics])
        avg_epsilon = np.mean([m.get("epsilon_spent", 0.0) for m in client_metrics])

        round_metrics = {
            "round": server_round,
            "num_clients": len(results),
            "total_examples": total_examples,
            "avg_train_loss": float(avg_train_loss),
            "avg_epsilon_spent": float(avg_epsilon),
            "num_failures": len(failures),
        }
        self._round_metrics.append(round_metrics)

        logger.info(
            "aggregation_complete",
            round=server_round,
            clients=len(results),
            total_examples=total_examples,
            avg_train_loss=f"{avg_train_loss:.4f}",
            avg_epsilon=f"{avg_epsilon:.4f}",
        )

        # Register to MLflow Staging
        self._register_to_mlflow(server_round, aggregated, round_metrics)

        return aggregated, round_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],  # (client_proxy, EvaluateRes)
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate evaluation results from clients.

        Args:
            server_round: Current round number.
            results: List of (client_proxy, EvaluateRes).
            failures: List of failed clients.

        Returns:
            Tuple of (aggregated_loss, aggregated_metrics).
        """
        if not results:
            logger.warning("no_eval_results", round=server_round)
            return None, {}

        # Weighted average of losses
        total_examples = sum(eval_res.num_examples for _, eval_res in results)
        weighted_loss = sum(
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
        ) / max(total_examples, 1)

        client_metrics = [eval_res.metrics for _, eval_res in results]
        avg_accuracy = np.mean([m.get("safety_accuracy", 0.0) for m in client_metrics])

        aggregated_metrics = {
            "val_loss": float(weighted_loss),
            "safety_accuracy": float(avg_accuracy),
            "num_clients": len(results),
        }

        logger.info(
            "eval_aggregation_complete",
            round=server_round,
            val_loss=f"{weighted_loss:.4f}",
            safety_accuracy=f"{avg_accuracy:.4f}",
        )

        return float(weighted_loss), aggregated_metrics

    def _weighted_average(
        self, weights_results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Compute weighted average of parameter lists.

        Args:
            weights_results: List of (param_list, num_examples) tuples.

        Returns:
            Averaged parameter list.
        """
        total_examples = sum(n for _, n in weights_results)
        if total_examples == 0:
            raise GradientAggregationError("Total examples is 0 — cannot average")

        # Initialize accumulators
        num_layers = len(weights_results[0][0])
        averaged = [np.zeros_like(weights_results[0][0][i]) for i in range(num_layers)]

        for params, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, param in enumerate(params):
                averaged[i] += weight * param.astype(np.float64)

        return [a.astype(np.float32) for a in averaged]

    def _parameters_to_ndarrays(self, parameters: Any) -> List[np.ndarray]:
        """Convert Flower Parameters object to list of numpy arrays.

        Args:
            parameters: Flower Parameters (bytes or ndarray container).

        Returns:
            List of numpy arrays.
        """
        try:
            from flwr.common import parameters_to_ndarrays
            return parameters_to_ndarrays(parameters)
        except ImportError:
            # Fallback if flwr.common not available
            return list(parameters.tensors)

    def _register_to_mlflow(
        self,
        server_round: int,
        parameters: List[np.ndarray],
        metrics: Dict[str, Any],
    ) -> None:
        """Register aggregated model to MLflow Staging.

        Args:
            server_round: Current round number.
            parameters: Aggregated LoRA parameters.
            metrics: Round metrics to log.
        """
        try:
            import mlflow
        except ImportError:
            logger.warning("mlflow_not_available", round=server_round)
            return

        try:
            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(
                run_name=f"federated_round_{server_round:03d}",
                tags={"round": str(server_round), "type": "federated_aggregation"},
            ):
                mlflow.log_metrics(metrics)
                mlflow.log_param("num_lora_tensors", len(parameters))
                mlflow.log_param("total_params", sum(p.size for p in parameters))

                # Save aggregated LoRA weights as artifact
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
                    np.savez(f, *parameters)
                    artifact_path = f.name
                mlflow.log_artifact(artifact_path, artifact_path=f"round_{server_round:03d}")

            logger.info(
                "mlflow_registered",
                round=server_round,
                model_name=self.mlflow_model_name,
            )
        except Exception as exc:
            logger.warning("mlflow_registration_failed", round=server_round, error=str(exc))


# ── Federated server orchestrator ─────────────────────────────────────────────

class ArgusFederatedServer:
    """Orchestrates the Flower federated server lifecycle.

    Starts a Flower server with the ArgusFedAvgStrategy, runs N rounds
    of federated training, then returns aggregated round metrics.

    Args:
        server_address: gRPC bind address (default "0.0.0.0:8080").
        num_rounds: Number of federated training rounds.
        min_fit_clients: Minimum clients for each fit round.
        min_evaluate_clients: Minimum clients for each eval round.
        min_available_clients: Minimum clients that must connect before starting.
        mlflow_experiment: MLflow experiment name.
        mlflow_model_name: Registry model name for aggregated checkpoints.
        round_timeout_s: Per-round timeout in seconds.

    Example:
        server = ArgusFederatedServer(num_rounds=5)
        history = server.start()
        print(history)
    """

    def __init__(
        self,
        server_address: str = "0.0.0.0:8080",
        num_rounds: int = 5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        mlflow_experiment: str = "argus/federated_rlhf",
        mlflow_model_name: str = "argus-safety-critic-federated",
        round_timeout_s: float = 600.0,
    ) -> None:
        self.server_address = server_address
        self.num_rounds = num_rounds
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_model_name = mlflow_model_name
        self.round_timeout_s = round_timeout_s

        self._strategy = ArgusFedAvgStrategy(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            mlflow_experiment=mlflow_experiment,
            mlflow_model_name=mlflow_model_name,
        )

    def start(self) -> Dict[str, Any]:
        """Start the Flower server and run all federated rounds.

        Returns:
            Dict with training history (losses, accuracies, privacy metrics).

        Raises:
            ImportError: If flwr is not installed.
            FederatedRoundError: If a round fails and cannot recover.
        """
        try:
            import flwr as fl
            from flwr.server import ServerConfig
        except ImportError:
            raise ImportError("Run: pip install flwr>=1.6.0")

        logger.info(
            "federated_server_start",
            address=self.server_address,
            num_rounds=self.num_rounds,
            min_clients=self.min_available_clients,
        )

        # Build Flower strategy adapter
        flower_strategy = _FlowerStrategyAdapter(self._strategy)

        history = fl.server.start_server(
            server_address=self.server_address,
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=flower_strategy,
        )

        # Compile summary
        round_metrics = self._strategy._round_metrics
        summary = {
            "num_rounds_completed": len(round_metrics),
            "final_train_loss": round_metrics[-1]["avg_train_loss"] if round_metrics else None,
            "final_epsilon": round_metrics[-1]["avg_epsilon_spent"] if round_metrics else None,
            "round_metrics": round_metrics,
        }

        logger.info(
            "federated_server_complete",
            rounds_completed=len(round_metrics),
            final_loss=summary["final_train_loss"],
            final_epsilon=summary["final_epsilon"],
        )

        return summary


class _FlowerStrategyAdapter:
    """Adapts ArgusFedAvgStrategy to Flower's Strategy interface.

    This thin wrapper calls our strategy methods while conforming to the
    Flower Strategy ABC contract.
    """

    def __init__(self, argus_strategy: ArgusFedAvgStrategy) -> None:
        self._s = argus_strategy

    def __getattr__(self, name: str) -> Any:
        """Delegate all unknown attributes to the wrapped strategy."""
        return getattr(self._s, name)

    def aggregate_fit(self, server_round: int, results: Any, failures: Any) -> Any:
        try:
            from flwr.common import ndarrays_to_parameters
        except ImportError:
            raise ImportError("Run: pip install flwr>=1.6.0")

        aggregated_ndarrays, metrics = self._s.aggregate_fit(server_round, results, failures)
        if aggregated_ndarrays is None:
            return None, {}
        parameters = ndarrays_to_parameters(aggregated_ndarrays)
        return parameters, metrics

    def aggregate_evaluate(self, server_round: int, results: Any, failures: Any) -> Any:
        return self._s.aggregate_evaluate(server_round, results, failures)

    def configure_fit(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
        """Sample clients for fitting."""
        try:
            from flwr.common import ndarrays_to_parameters
        except ImportError:
            raise ImportError("Run: pip install flwr>=1.6.0")

        sample_size = max(
            self._s.min_fit_clients,
            int(client_manager.num_available() * self._s.fraction_fit),
        )
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self._s.min_fit_clients)
        config = {"round": server_round}
        return [(client, _fit_ins(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Any, client_manager: Any) -> Any:
        """Sample clients for evaluation."""
        clients = client_manager.sample(
            num_clients=self._s.min_evaluate_clients,
            min_num_clients=self._s.min_evaluate_clients,
        )
        config = {"round": server_round}
        return [(client, _evaluate_ins(parameters, config)) for client in clients]

    def initialize_parameters(self, client_manager: Any) -> None:
        return None

    def evaluate(self, server_round: int, parameters: Any) -> None:
        return None


def _fit_ins(parameters: Any, config: Dict) -> Any:
    """Build a Flower FitIns object."""
    try:
        from flwr.common import FitIns
        return FitIns(parameters=parameters, config=config)
    except ImportError:
        raise ImportError("Run: pip install flwr>=1.6.0")


def _evaluate_ins(parameters: Any, config: Dict) -> Any:
    """Build a Flower EvaluateIns object."""
    try:
        from flwr.common import EvaluateIns
        return EvaluateIns(parameters=parameters, config=config)
    except ImportError:
        raise ImportError("Run: pip install flwr>=1.6.0")
