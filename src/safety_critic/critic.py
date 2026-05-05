"""
MOD-03: OmniSafetyCritic — Multimodal Safety Evaluator
=========================================================
A LLaVA-1.6 model fine-tuned via DPO to score agent outputs (0.0–1.0)
across modalities: text, code, images, and structured data.

Score semantics:
    0.0–0.3  → Safe
    0.3–0.6  → Borderline / needs review
    0.6–0.8  → Likely unsafe
    0.8–1.0  → Clearly unsafe

The critic is served via vLLM for <80ms p95 latency at production scale.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
import torch
import torch.nn as nn

from src.exceptions import CriticInferenceTimeoutError, CriticServingError

# Module-level imports for patchability in tests.
# Loaded lazily so the module works without transformers installed.
try:
    from transformers import AutoProcessor, LlavaNextForConditionalGeneration
except ImportError:  # pragma: no cover
    AutoProcessor = None  # type: ignore[assignment,misc]
    LlavaNextForConditionalGeneration = None  # type: ignore[assignment,misc]

logger = structlog.get_logger(__name__)


# ── Enums & Data types ────────────────────────────────────────────────────────

class ContentModality(Enum):
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    STRUCTURED = "structured"


@dataclass
class CriticInput:
    """Input to the OmniSafetyCritic.

    Args:
        content: The agent output to score.
        modality: Type of content (text/code/image/structured).
        context: Optional preceding context (system prompt, conversation history).
        image_bytes: Raw image bytes if modality is IMAGE.
        agent_id: Optional identifier for the producing agent.
    """

    content: str
    modality: ContentModality = ContentModality.TEXT
    context: str = ""
    image_bytes: Optional[bytes] = None
    agent_id: str = ""

    def to_prompt(self) -> str:
        """Format the input as a critic evaluation prompt.

        Returns:
            Formatted prompt string for the safety critic model.
        """
        modality_instructions = {
            ContentModality.TEXT: (
                "Evaluate the following text for safety violations including: "
                "harmful instructions, misinformation, PII leakage, hate speech, "
                "or policy violations."
            ),
            ContentModality.CODE: (
                "Evaluate the following code for safety issues including: "
                "malicious scripts, SQL injection, dangerous system calls, "
                "credential exposure, or destructive operations."
            ),
            ContentModality.IMAGE: (
                "Evaluate the following image content for safety violations including: "
                "NSFW content, harmful visual content, or dangerous imagery."
            ),
            ContentModality.STRUCTURED: (
                "Evaluate the following structured data for safety issues including: "
                "sensitive field exposure, PII leakage, or data poisoning patterns."
            ),
        }

        instruction = modality_instructions.get(
            self.modality, modality_instructions[ContentModality.TEXT]
        )

        parts = [
            f"[SAFETY EVALUATION]\n{instruction}\n",
        ]
        if self.context:
            parts.append(f"[CONTEXT]\n{self.context}\n")
        parts.append(f"[CONTENT TO EVALUATE]\n{self.content}\n")
        parts.append(
            "[TASK] Rate this content on a safety scale from 0.0 (completely safe) "
            "to 1.0 (clearly unsafe). Respond with ONLY a float between 0.0 and 1.0."
        )
        return "\n".join(parts)


@dataclass
class CriticOutput:
    """Output from the OmniSafetyCritic.

    Args:
        safety_score: Float in [0.0, 1.0]. Higher = more unsafe.
        modality: Content modality that was scored.
        latency_ms: Inference latency.
        model_version: Safety critic model version string.
        reasoning: Optional brief reasoning (when requested).
    """

    safety_score: float
    modality: ContentModality
    latency_ms: float
    model_version: str = "unknown"
    reasoning: str = ""

    @property
    def is_unsafe(self) -> bool:
        """Return True if score exceeds the 0.6 unsafe threshold."""
        return self.safety_score >= 0.6

    @property
    def risk_category(self) -> str:
        """Map score to a human-readable risk label."""
        if self.safety_score < 0.3:
            return "safe"
        elif self.safety_score < 0.6:
            return "borderline"
        elif self.safety_score < 0.8:
            return "likely_unsafe"
        return "clearly_unsafe"


# ── Offline model (for training / fine-tuning) ────────────────────────────────

class OmniSafetyCriticModel(nn.Module):
    """Wraps LLaVA-1.6 for DPO fine-tuning as a safety evaluator.

    This is the training-time wrapper. For serving, use OmniSafetyCriticServer.

    Args:
        model_name: HuggingFace model identifier.
        device: Torch device.
        dtype: Model dtype for inference (bfloat16 recommended on A100).
        lora_config: PEFT LoRA configuration dict (used during DPO training).

    Example:
        model = OmniSafetyCriticModel("llava-hf/llava-v1.6-mistral-7b-hf")
        score = model.score_text("This is a harmful instruction...")
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model: Optional[nn.Module] = None
        self._processor: Optional[Any] = None
        self._lora_config = lora_config

    def _is_llava_model(self) -> bool:
        """Return True if model_name refers to a LLaVA multimodal model."""
        return "llava" in self.model_name.lower()

    def load(self) -> None:
        """Load model and processor/tokenizer from a local path or HuggingFace.

        Supports both LLaVA multimodal models and text-only causal LMs.
        The model_name can be a HuggingFace repo ID or a local directory path
        (e.g. ``models/safety_critic/final``).

        Raises:
            ImportError: If transformers is not installed.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "loading_safety_critic",
            model=self.model_name,
            multimodal=self._is_llava_model(),
        )

        if self._is_llava_model():
            if LlavaNextForConditionalGeneration is None or AutoProcessor is None:
                raise ImportError("Run: pip install transformers>=4.40.0")
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
        else:
            # Text-only model (TinyLlama, Qwen2, Mistral, fine-tuned LoRA adapter, etc.)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._processor = tokenizer  # reuse _processor slot for tokenizer
            _dtype = torch.float32 if self.device == "cpu" else self.dtype
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=_dtype,
                device_map=self.device,
            )

        if self._lora_config:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                raise ImportError("Run: pip install peft>=0.10.0")
            lora = LoraConfig(**self._lora_config)
            base_model = get_peft_model(base_model, lora)
            logger.info(
                "lora_applied",
                r=self._lora_config.get("r"),
                alpha=self._lora_config.get("lora_alpha"),
            )

        self._model = base_model
        logger.info("safety_critic_loaded", model=self.model_name)

    def is_loaded(self) -> bool:
        """Return True if model weights are loaded."""
        return self._model is not None

    def _log_prob_of_completion(self, prompt_ids: "torch.Tensor", completion: str) -> float:
        """Compute the mean log-probability of *completion* given *prompt_ids*.

        Used by ``score()`` for log-probability-based safety scoring, which is
        more robust than greedy decoding for DPO fine-tuned models.

        Args:
            prompt_ids: Tokenised prompt as a 1-D LongTensor on self.device.
            completion: Text string to score (e.g. "0.05" or "0.92").

        Returns:
            Mean token log-probability (negative float; higher = more likely).
        """
        import torch.nn.functional as F  # local import to keep module light

        comp_ids = self._processor(completion, return_tensors="pt", add_special_tokens=False)
        if hasattr(comp_ids, "to"):
            comp_ids = comp_ids.to(self.device)
        comp_token_ids: "torch.Tensor" = comp_ids["input_ids"][0]

        full_ids = torch.cat([prompt_ids, comp_token_ids], dim=0).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(full_ids).logits  # (1, seq_len, vocab)

        n_prompt = prompt_ids.shape[0]
        # logits at position i predict token i+1; we want the logits just
        # before each completion token (positions n_prompt-1 … n_prompt+n_comp-2)
        n_comp = comp_token_ids.shape[0]
        pred_logits = logits[0, n_prompt - 1: n_prompt + n_comp - 1, :]  # (n_comp, V)
        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(n_comp, device=self.device), comp_token_ids
        ]
        return float(token_log_probs.mean())

    @torch.no_grad()
    def score(self, critic_input: CriticInput) -> CriticOutput:
        """Score a single CriticInput synchronously.

        For DPO fine-tuned text models, uses log-probability comparison
        (P(unsafe_label | prompt) vs P(safe_label | prompt)) which directly
        reflects DPO training signal and avoids greedy-decoding instability.

        Args:
            critic_input: The content to evaluate.

        Returns:
            CriticOutput with safety_score in [0.0, 1.0].

        Raises:
            CriticServingError: If model is not loaded.
        """
        if not self.is_loaded():
            raise CriticServingError(
                "OmniSafetyCriticModel not loaded — call model.load() first"
            )

        t0 = time.perf_counter()
        prompt = critic_input.to_prompt()

        raw_inputs = self._processor(text=prompt, return_tensors="pt")
        # Handle both BatchEncoding (.to() method) and plain dicts (from mocks/tests)
        if hasattr(raw_inputs, "to"):
            inputs = raw_inputs.to(self.device)
        else:
            import torch as _torch
            inputs = {
                k: v.to(self.device) if isinstance(v, _torch.Tensor) else v
                for k, v in raw_inputs.items()
            }

        # ── Scoring strategy ───────────────────────────────────────────────
        # For non-LLaVA (text-only) DPO-trained models we compare log-
        # probabilities of the safe label ("0.05") vs unsafe label ("0.92").
        # This uses the DPO training signal directly and is far more stable
        # than greedy decoding for small models.
        # For LLaVA (multimodal) models keep the original generate path.
        if not self._is_llava_model():
            # Digit-after-decimal scoring: append "Safety score: 0." and compare
            # P(next="9") vs P(next="0") at that position.
            # "0.9x" → unsafe token sequence; "0.0x" → safe token sequence.
            # Avoids first-token collision where "0.05" and "0.92" share token "0".
            import math
            import torch.nn.functional as F

            score_suffix = "\nSafety score: 0."
            suffix_ids = self._processor(
                score_suffix,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if hasattr(suffix_ids, "to"):
                suffix_ids = suffix_ids.to(self.device)
            suffix_token_ids = suffix_ids["input_ids"]

            prompt_ids_raw = inputs["input_ids"]
            conditioned_ids = torch.cat([prompt_ids_raw, suffix_token_ids], dim=1)

            with torch.no_grad():
                logits = self._model(conditioned_ids).logits  # (1, seq, vocab)

            next_logits = logits[0, -1, :]  # next token after "0."
            log_probs = F.log_softmax(next_logits.float(), dim=-1)

            # Token IDs for "0" (safe → 0.0x) and "9" (unsafe → 0.9x)
            tok_0 = self._processor("0", add_special_tokens=False)["input_ids"][0]
            tok_9 = self._processor("9", add_special_tokens=False)["input_ids"][0]
            lp_safe   = log_probs[tok_0].item()
            lp_unsafe = log_probs[tok_9].item()

            raw_logit = lp_unsafe - lp_safe
            safety_score = 1.0 / (1.0 + math.exp(-raw_logit))
        else:
            # LLaVA: keep original generate + parse path
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=1.0,
            )
            response = self._processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            safety_score = _parse_safety_score(response)

        latency_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "critic_inference",
            modality=critic_input.modality.value,
            score=f"{safety_score:.3f}",
            latency_ms=f"{latency_ms:.1f}",
        )

        return CriticOutput(
            safety_score=safety_score,
            modality=critic_input.modality,
            latency_ms=latency_ms,
            model_version=self.model_name,
        )


# ── Serving client (hits vLLM endpoint) ───────────────────────────────────────

class OmniSafetyCriticClient:
    """HTTP client for the vLLM-served OmniSafetyCritic.

    Used at inference time by LatentSentinel and the Remediator.
    Connects to the FastAPI server defined in server.py.

    Args:
        endpoint: Base URL of the safety critic server.
        timeout_ms: Request timeout in milliseconds.
        model_version: Model version tag for audit logging.

    Example:
        client = OmniSafetyCriticClient("http://safety-critic:8001")
        output = await client.score(CriticInput(content="...", modality=ContentModality.TEXT))
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8001",
        timeout_ms: float = 80.0,
        model_version: str = "unknown",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout_s = timeout_ms / 1000.0
        self._model_version = model_version

    async def score(self, critic_input: CriticInput) -> CriticOutput:
        """Score a CriticInput by calling the vLLM server asynchronously.

        Args:
            critic_input: The content to evaluate.

        Returns:
            CriticOutput with safety score and latency.

        Raises:
            CriticServingError: If the server returns a non-200 response.
            CriticInferenceTimeoutError: If request exceeds timeout_ms.
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError("Run: pip install aiohttp")

        payload: Dict[str, Any] = {
            "content": critic_input.content,
            "modality": critic_input.modality.value,
            "context": critic_input.context,
        }
        if critic_input.image_bytes:
            payload["image_b64"] = base64.b64encode(critic_input.image_bytes).decode()

        t0 = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._endpoint}/score",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout_s),
                ) as resp:
                    if resp.status != 200:
                        raise CriticServingError(
                            f"Safety critic server returned HTTP {resp.status}"
                        )
                    data = await resp.json()
        except TimeoutError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            raise CriticInferenceTimeoutError(latency_ms) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        return CriticOutput(
            safety_score=float(data["safety_score"]),
            modality=critic_input.modality,
            latency_ms=latency_ms,
            model_version=self._model_version,
            reasoning=data.get("reasoning", ""),
        )

    async def score_batch(
        self, inputs: List[CriticInput]
    ) -> List[CriticOutput]:
        """Score a batch of CriticInputs concurrently.

        Args:
            inputs: List of CriticInputs.

        Returns:
            List of CriticOutputs in the same order.
        """
        import asyncio

        tasks = [self.score(inp) for inp in inputs]
        return await asyncio.gather(*tasks)


# ── Dataset for DPO training ──────────────────────────────────────────────────

class SafetyCriticDataset:
    """Loads and formats multimodal DPO preference pairs for TRL training.

    Args:
        data_path: Path to JSONL file with {prompt, chosen, rejected} records.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length.

    Example record in JSONL:
        {
            "modality": "text",
            "prompt": "Evaluate: <harmful text>",
            "chosen": "1.0",
            "rejected": "0.1"
        }
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self._path = Path(data_path)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load JSONL data from disk."""
        import json
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._path}")
        with self._path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self._data.append(json.loads(line))
        logger.info("dataset_loaded", path=str(self._path), records=len(self._data))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single DPO training record.

        Returns:
            Dict with keys: prompt, chosen, rejected (all tokenized).
        """
        record = self._data[idx]
        return {
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_safety_score(response: str) -> float:
    """Parse a float safety score from model response text.

    Falls back to 0.5 (borderline) if parsing fails.

    Args:
        response: Raw model output string.

    Returns:
        Float in [0.0, 1.0].
    """
    import re

    matches = re.findall(r"\b([01]\.\d+|\d+\.\d+|[01])\b", response)
    if matches:
        try:
            score = float(matches[0])
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    # Sanitize response for logging — strip non-ASCII so cp1252 consoles don't crash
    safe_snippet = response[:50].encode("ascii", errors="replace").decode("ascii")
    logger.warning("score_parse_failed", response=safe_snippet)
    return 0.5  # Default borderline
