from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog

from src.exceptions import FederatedRoundError, GradientAggregationError

logger = structlog.get_logger(__name__)

class ArgusFedAvgStrategy:
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
        results: List[Tuple[Any, Any]],  
        failures: List[Any],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, Any]]:

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

        weights_results = []
        for client_proxy, fit_res in results:
            params = self._parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            weights_results.append((params, num_examples))

        try:
            aggregated = self._weighted_average(weights_results)
        except Exception as exc:
            raise GradientAggregationError(
                f"Weighted averaging failed at round {server_round}: {exc}"
            ) from exc

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

        self._register_to_mlflow(server_round, aggregated, round_metrics)

        return aggregated, round_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[Any, Any]],  
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            logger.warning("no_eval_results", round=server_round)
            return None, {}

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

        total_examples = sum(n for _, n in weights_results)
        if total_examples == 0:
            raise GradientAggregationError("Total examples is 0 — cannot average")

        num_layers = len(weights_results[0][0])
        averaged = [np.zeros_like(weights_results[0][0][i]) for i in range(num_layers)]

        for params, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, param in enumerate(params):
                averaged[i] += weight * param.astype(np.float64)

        return [a.astype(np.float32) for a in averaged]

    def _parameters_to_ndarrays(self, parameters: Any) -> List[np.ndarray]:
        try:
            from flwr.common import parameters_to_ndarrays
            return parameters_to_ndarrays(parameters)
        except ImportError:
            return list(parameters.tensors)

    def _register_to_mlflow(
        self,
        server_round: int,
        parameters: List[np.ndarray],
        metrics: Dict[str, Any],
    ) -> None:
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


class ArgusFederatedServer:
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

        flower_strategy = _FlowerStrategyAdapter(self._strategy)

        history = fl.server.start_server(
            server_address=self.server_address,
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=flower_strategy,
        )

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
    try:
        from flwr.common import FitIns
        return FitIns(parameters=parameters, config=config)
    except ImportError:
        raise ImportError("Run: pip install flwr>=1.6.0")


def _evaluate_ins(parameters: Any, config: Dict) -> Any:
    try:
        from flwr.common import EvaluateIns
        return EvaluateIns(parameters=parameters, config=config)
    except ImportError:
        raise ImportError("Run: pip install flwr>=1.6.0")
