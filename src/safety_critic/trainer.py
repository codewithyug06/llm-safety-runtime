"""
MOD-03: OmniSafetyCritic DPO Training Loop
============================================
Wraps TRL's DPOTrainer for fine-tuning LLaVA-1.6 as a multimodal safety critic
using Direct Preference Optimization on adversarially constructed preference pairs.

Key design decisions:
- DPO over PPO: eliminates separate reward model, more stable training
- LoRA r=16, alpha=32: parameter-efficient, preserves base model capabilities
- bfloat16 + gradient checkpointing: fits 7B model on 4×A100 80GB
- W&B + MLflow: dual tracking for experiment and registry management

Run via: python scripts/train_safety_critic.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import torch

from src.exceptions import SafetyCriticError

# ── Compatibility shim ────────────────────────────────────────────────────────
# TRL >= 1.0 imports FSDPModule from torch.distributed.fsdp, but PyTorch
# moved it to torch.distributed._composable.fsdp (FSDP2).  Patch the legacy
# path so the import resolves without downgrading TRL.
try:
    import torch.distributed.fsdp as _fsdp_legacy
    if not hasattr(_fsdp_legacy, "FSDPModule"):
        from torch.distributed._composable.fsdp import FSDPModule as _FSDPModule
        _fsdp_legacy.FSDPModule = _FSDPModule  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass  # non-distributed environment — safe to ignore

logger = structlog.get_logger(__name__)


class SafetyCriticTrainer:
    """Orchestrates DPO fine-tuning of the OmniSafetyCritic.

    Args:
        model_name: HuggingFace base model (LLaVA-1.6).
        output_dir: Directory to save checkpoints and final model.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        dpo_beta: KL penalty coefficient for DPO loss.
        batch_size: Per-device training batch size.
        grad_accum_steps: Gradient accumulation steps.
        num_epochs: Training epochs.
        learning_rate: AdamW learning rate.
        mlflow_experiment: MLflow experiment name.
        wandb_project: W&B project name.

    Example:
        trainer = SafetyCriticTrainer(
            model_name="llava-hf/llava-v1.6-mistral-7b-hf",
            output_dir="models/safety_critic",
        )
        trainer.train(train_dataset, eval_dataset)
        trainer.register_to_mlflow("argus-safety-critic-v1")
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        output_dir: str = "models/safety_critic",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dpo_beta: float = 0.1,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        max_length: int = 1024,
        max_prompt_length: int = 512,
        mlflow_experiment: str = "argus/safety_critic/dpo",
        wandb_project: str = "argus-safety-critic",
        load_in_4bit: bool = False,
        skip_wandb: bool = False,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.dpo_beta = dpo_beta
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.mlflow_experiment = mlflow_experiment
        self.wandb_project = wandb_project
        self.load_in_4bit = load_in_4bit
        self.skip_wandb = skip_wandb

        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._trainer: Optional[Any] = None

    def _build_lora_config(self) -> Any:
        """Build PEFT LoRA configuration.

        Returns:
            LoraConfig instance.
        """
        try:
            from peft import LoraConfig, TaskType
        except ImportError:
            raise ImportError("Run: pip install peft>=0.10.0")

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _build_training_args(self, report_to: list | None = None) -> Any:
        """Build HuggingFace TrainingArguments for DPO.

        Returns:
            DPOConfig (extends TrainingArguments) instance.
        """
        try:
            from trl import DPOConfig
        except ImportError:
            raise ImportError("Run: pip install trl>=0.8.0")

        import inspect as _inspect
        _dpo_params = set(_inspect.signature(DPOConfig.__init__).parameters)

        kwargs: Dict[str, Any] = dict(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum_steps,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Windows: 0 avoids multiprocessing errors
            max_grad_norm=1.0,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=report_to or ["none"],
            run_name=f"dpo-{self.model_name.split('/')[-1]}",
            # DPO-specific
            beta=self.dpo_beta,
            max_length=self.max_length,
            loss_type="sigmoid",
            remove_unused_columns=False,
        )
        # max_prompt_length was removed in TRL >= 1.x; only pass if accepted
        if "max_prompt_length" in _dpo_params:
            kwargs["max_prompt_length"] = self.max_prompt_length

        return DPOConfig(**kwargs)

    def _is_llava_model(self) -> bool:
        """Return True if the model is a LLaVA multimodal model."""
        return "llava" in self.model_name.lower()

    def load_model(self) -> None:
        """Load base model with LoRA adapters.

        Supports both LLaVA multimodal models and text-only causal LMs.
        Uses 4-bit NF4 QLoRA when ``load_in_4bit=True`` (fits 6 GB VRAM GPUs).

        Sets self._model and self._tokenizer.

        Raises:
            SafetyCriticError: If model loading fails.
        """
        try:
            from peft import get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError:
            raise ImportError("Run: pip install transformers peft bitsandbytes")

        logger.info(
            "loading_base_model",
            model=self.model_name,
            load_in_4bit=self.load_in_4bit,
            multimodal=self._is_llava_model(),
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 4-bit QLoRA: NF4 quantisation for GPUs with <=8 GB VRAM
        bnb_config: Optional[BitsAndBytesConfig] = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("qlora_4bit_enabled", quant_type="nf4", compute_dtype="bfloat16")

        if self._is_llava_model():
            from transformers import LlavaNextForConditionalGeneration
            base = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
        else:
            # Text-only causal LM (TinyLlama, Qwen2, Mistral, etc.)
            _dtype = torch.float32 if not torch.cuda.is_available() else torch.bfloat16
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=_dtype,
                device_map="auto",
                quantization_config=bnb_config,
            )

        if self.load_in_4bit:
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=True
            )

        lora_config = self._build_lora_config()
        self._model = get_peft_model(base, lora_config)
        self._model.print_trainable_parameters()
        logger.info("model_loaded_with_lora", model=self.model_name)

    def train(self, train_dataset: Any, eval_dataset: Optional[Any] = None) -> None:
        """Run DPO fine-tuning.

        Args:
            train_dataset: Dataset of preference pairs (prompt/chosen/rejected).
            eval_dataset: Optional evaluation dataset.

        Raises:
            SafetyCriticError: If training loop fails.
        """
        if self._model is None:
            self.load_model()

        try:
            import mlflow
            from trl import DPOTrainer
        except ImportError:
            raise ImportError("Run: pip install trl mlflow")

        _wandb_active = False
        if not self.skip_wandb:
            try:
                import wandb
                wandb.init(project=self.wandb_project, name="dpo-safety-critic")
                _wandb_active = True
            except Exception:
                pass

        try:
            mlflow.set_experiment(self.mlflow_experiment)
            _mlflow_active = True
        except Exception:
            _mlflow_active = False

        # Disable W&B/MLflow reporting in TrainingArguments if not available
        _report_to = []
        if _wandb_active:
            _report_to.append("wandb")
        if _mlflow_active:
            _report_to.append("mlflow")
        if not _report_to:
            _report_to = ["none"]

        training_args = self._build_training_args(report_to=_report_to)

        logger.info(
            "dpo_training_start",
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum_steps,
            effective_batch=self.batch_size * self.grad_accum_steps,
            beta=self.dpo_beta,
            load_in_4bit=self.load_in_4bit,
        )

        run_ctx = mlflow.start_run() if _mlflow_active else None
        try:
            if _mlflow_active:
                mlflow.log_params({
                    "model": self.model_name,
                    "lora_r": self.lora_r,
                    "dpo_beta": self.dpo_beta,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "load_in_4bit": self.load_in_4bit,
                })

            self._trainer = DPOTrainer(
                model=self._model,
                ref_model=None,  # None → implicit reference (LoRA base)
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self._tokenizer,
            )

            train_result = self._trainer.train()

            if _mlflow_active:
                mlflow.log_metrics({
                    "train_loss": train_result.training_loss,
                    "train_runtime_s": train_result.metrics.get("train_runtime", 0),
                })

            # Save final model
            self._trainer.save_model(str(self.output_dir / "final"))
            logger.info("training_complete", output_dir=str(self.output_dir / "final"))
        finally:
            if _mlflow_active and run_ctx is not None:
                mlflow.end_run()
            if _wandb_active:
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass

    def evaluate(self, eval_dataset: Any) -> Dict[str, float]:
        """Evaluate on a held-out dataset.

        Args:
            eval_dataset: Evaluation dataset.

        Returns:
            Dict of evaluation metrics.
        """
        if self._trainer is None:
            raise SafetyCriticError("Call train() before evaluate()")
        metrics = self._trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("evaluation_complete", metrics=metrics)
        return metrics

    def register_to_mlflow(
        self,
        model_name: str = "argus-safety-critic",
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register the trained model to MLflow model registry.

        Args:
            model_name: Registry model name.
            tags: Optional key-value tags for the model version.

        Returns:
            MLflow model URI string.
        """
        try:
            import mlflow
        except ImportError:
            raise ImportError("Run: pip install mlflow")

        model_path = str(self.output_dir / "final")
        uri = mlflow.pyfunc.log_model(
            artifact_path="safety_critic",
            python_model=None,
            artifacts={"model_dir": model_path},
            registered_model_name=model_name,
        ).model_uri

        logger.info("model_registered", model_name=model_name, uri=uri)
        return uri
