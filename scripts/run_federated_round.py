"""
MOD-04: Federated RLHF Round Runner
=====================================
Starts a Flower server + N simulated client nodes (on separate threads)
to demonstrate one or more federated DPO training rounds locally.

In production, clients run on separate machines / GKE pods.
This script simulates the full flow on a single machine for testing.

Run with:
    python scripts/run_federated_round.py
    python scripts/run_federated_round.py --num-clients 3 --num-rounds 5
    python scripts/run_federated_round.py --server-address localhost:9090 --no-dp
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import List

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.federated_rlhf.client import ArgusFederatedClient
from src.federated_rlhf.privacy import DPSGDOpacusWrapper
from src.federated_rlhf.server import ArgusFederatedServer

logger = structlog.get_logger(__name__)


def _start_client(
    client_id: str,
    server_address: str,
    model_name: str,
    train_data_path: str,
    val_data_path: str,
    use_dp: bool,
    epsilon_budget: float,
    device: str,
) -> None:
    """Start a single federated client and connect to server.

    Args:
        client_id: Unique client identifier.
        server_address: Server address (host:port).
        model_name: HuggingFace model name.
        train_data_path: Path to local training JSONL.
        val_data_path: Path to local validation JSONL.
        use_dp: Whether to apply DP-SGD.
        epsilon_budget: Privacy budget for DP-SGD.
        device: Torch device string.
    """
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Run: pip install flwr>=1.6.0")

    dp_wrapper = DPSGDOpacusWrapper(epsilon_budget=epsilon_budget) if use_dp else None

    client = ArgusFederatedClient(
        client_id=client_id,
        model_name=model_name,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        dp_wrapper=dp_wrapper,
        device=device,
    )

    logger.info("client_connecting", client_id=client_id, server=server_address)

    # Wrap in Flower NumPyClient adapter
    flower_client = _FlowerClientAdapter(client)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=flower_client,
    )


class _FlowerClientAdapter:
    """Adapter making ArgusFederatedClient conform to Flower NumPyClient."""

    def __init__(self, argus_client: ArgusFederatedClient) -> None:
        self._client = argus_client

    def get_parameters(self, config):
        return self._client.get_parameters(config)

    def fit(self, parameters, config):
        return self._client.fit(parameters, config)

    def evaluate(self, parameters, config):
        return self._client.evaluate(parameters, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARGUS federated RLHF round")
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument(
        "--model",
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="HuggingFace model name",
    )
    parser.add_argument("--train-data", default="data/safety_critic/train.jsonl")
    parser.add_argument("--val-data", default="data/safety_critic/val.jsonl")
    parser.add_argument("--no-dp", action="store_true", help="Disable DP-SGD")
    parser.add_argument("--epsilon-budget", type=float, default=3.0)
    parser.add_argument("--mlflow-experiment", default="argus/federated_rlhf")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    use_dp = not args.no_dp

    logger.info(
        "federated_round_start",
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        dp_enabled=use_dp,
        epsilon_budget=args.epsilon_budget,
    )

    # Start server in a background thread (with a short delay to allow binding)
    server = ArgusFederatedServer(
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        mlflow_experiment=args.mlflow_experiment,
    )

    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Give the server a moment to bind
    time.sleep(2.0)

    # Start clients in parallel threads
    client_threads: List[threading.Thread] = []
    for i in range(args.num_clients):
        t = threading.Thread(
            target=_start_client,
            kwargs={
                "client_id": f"client_{i}",
                "server_address": args.server_address,
                "model_name": args.model,
                "train_data_path": args.train_data,
                "val_data_path": args.val_data,
                "use_dp": use_dp,
                "epsilon_budget": args.epsilon_budget,
                "device": args.device,
            },
            daemon=True,
        )
        client_threads.append(t)
        t.start()
        # Stagger client starts slightly
        time.sleep(0.5)

    # Wait for all clients to complete
    for t in client_threads:
        t.join(timeout=3600)

    # Wait for server to finish
    server_thread.join(timeout=60)

    logger.info("federated_round_complete", rounds=args.num_rounds)

    print(f"\n=== Federated Round Complete ===")
    print(f"  Rounds     : {args.num_rounds}")
    print(f"  Clients    : {args.num_clients}")
    print(f"  DP-SGD     : {'enabled (ε≤' + str(args.epsilon_budget) + ')' if use_dp else 'disabled'}")
    print(f"  MLflow exp : {args.mlflow_experiment}")


if __name__ == "__main__":
    main()
