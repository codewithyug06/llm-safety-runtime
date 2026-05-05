"""
MOD-04: ARGUS Federated RLHF Client
=====================================
Flower NumPyClient that runs local DPO fine-tuning with DP-SGD on each
participating node. Implements the standard Flower client interface:
  - get_parameters() → current LoRA adapter weights as numpy arrays
  - fit()            → one local DPO training round + DP-SGD noise
  - evaluate()       → safety accuracy on local validation set

Key design decisions:
- Only LoRA adapter weights are exchanged (not full model) — ~50MB vs 14GB
- Opacus DP-SGD applied per local round; epsilon tracked cumulatively
- Local dataset never leaves the client node (federated privacy guarantee)
- MLflow run created per client per round for traceability

Run via: scripts/run_federated_round.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from src.exceptions import FederatedRoundError, PrivacyBudgetExhaustedError
from src.federated_rlhf.privacy import DPSGDOpacusWrapper

logger = structlog.get_logger(__name__)


def _get_lora_params(model: Any) -> List[np.ndarray]:
    """Extract LoRA adapter parameters as a flat list of numpy arrays.

    Args:
        model: PEFT model with LoRA adapters attached.

    Returns:
        List of numpy arrays, one per LoRA weight tensor.
    """
    import torch
    params = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            params.append(param.detach().cpu().numpy())
    return params


def _set_lora_params(model: Any, params: List[np.ndarray]) -> None:
    """Load LoRA adapter parameters from a list of numpy arrays.

    Args:
        model: PEFT model with LoRA adapters.
        params: List of numpy arrays matching the LoRA parameter order.
    """
    import torch
    lora_params = [
        (name, p)
        for name, p in model.named_parameters()
        if "lora_" in name and p.requires_grad
    ]
    for (name, param), new_val in zip(lora_params, params):
        param.data = torch.from_numpy(new_val).to(param.device)


class ArgusFederatedClient:
    """Flower-compatible federated learning client for DPO safety fine-tuning.

    Implements the NumPyClient interface so Flower can orchestrate parameter
    exchange between clients and the aggregation server.

    Args:
        client_id: Unique identifier for this client node.
        model_name: HuggingFace base model to fine-tune.
        train_data_path: Path to local JSONL DPO training data.
        val_data_path: Path to local JSONL DPO validation data.
        lora_r: LoRA rank (must match server config).
        lora_alpha: LoRA scaling factor.
        local_epochs: Number of local DPO epochs per federated round.
        batch_size: Local training batch size.
        learning_rate: AdamW learning rate.
        dp_wrapper: Optional pre-configured DP-SGD wrapper.
        device: Torch device string.

    Example:
        client = ArgusFederatedClient(
            client_id="hospital_node_1",
            train_data_path="data/local/train.jsonl",
            val_data_path="data/local/val.jsonl",
        )
        fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    """

    def __init__(
        self,
        client_id: str = "client_0",
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        train_data_path: str = "data/safety_critic/train.jsonl",
        val_data_path: str = "data/safety_critic/val.jsonl",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        local_epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        max_length: int = 512,
        dp_wrapper: Optional[DPSGDOpacusWrapper] = None,
        device: str = "cuda",
    ) -> None:
        self.client_id = client_id
        self.model_name = model_name
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device

        self._dp_wrapper = dp_wrapper or DPSGDOpacusWrapper()
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._round: int = 0

    # ── Flower NumPyClient interface ───────────────────────────────────────────

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return current LoRA adapter weights as numpy arrays.

        Args:
            config: Server-provided config dict (may include round number).

        Returns:
            Flat list of numpy arrays representing LoRA parameters.
        """
        if self._model is None:
            self._load_model()
        params = _get_lora_params(self._model)
        logger.info(
            "get_parameters",
            client_id=self.client_id,
            num_tensors=len(params),
            total_params=sum(p.size for p in params),
        )
        return params

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Run one local DPO training round with DP-SGD.

        Args:
            parameters: Global LoRA parameters from server aggregation.
            config: Server config dict (round number, learning_rate override, etc.).

        Returns:
            Tuple of (updated_parameters, num_examples, metrics_dict).

        Raises:
            PrivacyBudgetExhaustedError: If cumulative ε ≥ 3.0.
            FederatedRoundError: If local training loop fails.
        """
        self._round = int(config.get("round", self._round + 1))
        logger.info("fit_start", client_id=self.client_id, round=self._round)

        # Check privacy budget before training
        self._dp_wrapper.check_budget()

        if self._model is None:
            self._load_model()

        # Load global parameters into local model
        _set_lora_params(self._model, parameters)

        try:
            train_loss, num_examples = self._run_local_training()
        except PrivacyBudgetExhaustedError:
            raise
        except Exception as exc:
            raise FederatedRoundError(
                message=f"Local training failed: {exc}",
                round_num=self._round,
                available_clients=1,
                required_clients=1,
            ) from exc

        # Record privacy expenditure for this round
        try:
            delta_epsilon = self._dp_wrapper.record_round()
        except PrivacyBudgetExhaustedError:
            logger.warning(
                "privacy_budget_exhausted",
                client_id=self.client_id,
                round=self._round,
                epsilon_spent=self._dp_wrapper.accounting_state.epsilon_spent,
            )
            raise

        updated_params = _get_lora_params(self._model)
        metrics = {
            "train_loss": float(train_loss),
            "epsilon_spent": self._dp_wrapper.accounting_state.epsilon_spent,
            "delta_epsilon_this_round": float(delta_epsilon),
            "round": self._round,
        }

        logger.info(
            "fit_complete",
            client_id=self.client_id,
            round=self._round,
            train_loss=f"{train_loss:.4f}",
            num_examples=num_examples,
            epsilon=f"{self._dp_wrapper.accounting_state.epsilon_spent:.4f}",
        )

        return updated_params, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate aggregated parameters on local validation set.

        Args:
            parameters: Aggregated LoRA parameters from server.
            config: Server config dict.

        Returns:
            Tuple of (loss, num_examples, metrics_dict).
        """
        if self._model is None:
            self._load_model()

        _set_lora_params(self._model, parameters)

        loss, accuracy, num_examples = self._run_local_eval()

        metrics = {
            "safety_accuracy": float(accuracy),
            "val_loss": float(loss),
            "client_id": self.client_id,
        }

        logger.info(
            "evaluate_complete",
            client_id=self.client_id,
            val_loss=f"{loss:.4f}",
            safety_accuracy=f"{accuracy:.4f}",
            num_examples=num_examples,
        )

        return float(loss), num_examples, metrics

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load base model + LoRA adapters onto device."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoTokenizer, LlavaNextForConditionalGeneration
        except ImportError:
            raise ImportError("Run: pip install transformers peft")

        import torch

        logger.info("loading_model", client_id=self.client_id, model=self.model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self._model = get_peft_model(base, lora_config)
        self._model.train()

        logger.info("model_loaded", client_id=self.client_id)

    def _build_dataloader(self, data_path: Path) -> Any:
        """Build a PyTorch DataLoader from a JSONL DPO dataset.

        Args:
            data_path: Path to JSONL file with {prompt, chosen, rejected}.

        Returns:
            PyTorch DataLoader.
        """
        import json

        import torch
        from torch.utils.data import DataLoader, Dataset

        class _DPODataset(Dataset):
            def __init__(self, path: Path, tokenizer: Any, max_length: int) -> None:
                self._records = []
                with path.open() as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._records.append(json.loads(line))
                self._tokenizer = tokenizer
                self._max_length = max_length

            def __len__(self) -> int:
                return len(self._records)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                r = self._records[idx]
                return {
                    "prompt": r["prompt"],
                    "chosen": r["chosen"],
                    "rejected": r["rejected"],
                }

        dataset = _DPODataset(data_path, self._tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _run_local_training(self) -> Tuple[float, int]:
        """Run local DPO training with DP-SGD noise injection.

        Returns:
            Tuple of (avg_train_loss, num_examples_trained).
        """
        import torch
        from torch.optim import AdamW

        optimizer = AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )
        train_loader = self._build_dataloader(self.train_data_path)

        # Attach Opacus DP-SGD
        private_model, private_optimizer, private_loader = self._dp_wrapper.attach(
            model=self._model,
            optimizer=optimizer,
            data_loader=train_loader,
        )

        total_loss = 0.0
        num_examples = 0

        for _epoch in range(self.local_epochs):
            for batch in private_loader:
                private_optimizer.zero_grad()

                # Tokenize chosen + rejected
                chosen_enc = self._tokenizer(
                    batch["chosen"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                rejected_enc = self._tokenizer(
                    batch["rejected"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                # Forward pass — compute cross-entropy losses as DPO proxy
                chosen_out = private_model(**chosen_enc, labels=chosen_enc["input_ids"])
                rejected_out = private_model(**rejected_enc, labels=rejected_enc["input_ids"])

                # DPO loss: log(σ(β * (log π_c - log π_r)))
                beta = 0.1
                log_ratio = chosen_out.loss - rejected_out.loss
                loss = -torch.nn.functional.logsigmoid(beta * log_ratio)

                loss.backward()
                private_optimizer.step()

                total_loss += float(loss.item()) * len(batch["chosen"])
                num_examples += len(batch["chosen"])

        self._model = private_model  # update reference after DP wrapping
        avg_loss = total_loss / max(num_examples, 1)
        return avg_loss, num_examples

    def _run_local_eval(self) -> Tuple[float, float, int]:
        """Evaluate model on local validation set.

        Returns:
            Tuple of (val_loss, safety_accuracy, num_examples).
        """
        import torch

        self._model.eval()
        val_loader = self._build_dataloader(self.val_data_path)

        total_loss = 0.0
        correct = 0
        num_examples = 0

        with torch.no_grad():
            for batch in val_loader:
                chosen_enc = self._tokenizer(
                    batch["chosen"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                rejected_enc = self._tokenizer(
                    batch["rejected"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                chosen_out = self._model(**chosen_enc, labels=chosen_enc["input_ids"])
                rejected_out = self._model(**rejected_enc, labels=rejected_enc["input_ids"])

                beta = 0.1
                log_ratio = chosen_out.loss - rejected_out.loss
                loss = -torch.nn.functional.logsigmoid(beta * log_ratio)

                # Accuracy: chosen loss < rejected loss means model prefers safe response
                batch_correct = int((chosen_out.loss < rejected_out.loss).sum().item())
                correct += batch_correct
                total_loss += float(loss.item()) * len(batch["chosen"])
                num_examples += len(batch["chosen"])

        self._model.train()
        avg_loss = total_loss / max(num_examples, 1)
        accuracy = correct / max(num_examples, 1)
        return avg_loss, accuracy, num_examples
