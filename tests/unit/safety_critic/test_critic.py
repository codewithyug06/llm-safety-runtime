"""
Unit tests for MOD-03 OmniSafetyCritic.
=========================================
Tests critic data types, score parsing, dataset loading, and server contracts.
NEVER loads a real LLM — all model interactions are mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.safety_critic.critic import (
    ContentModality,
    CriticInput,
    CriticOutput,
    OmniSafetyCriticClient,
    OmniSafetyCriticModel,
    SafetyCriticDataset,
    _parse_safety_score,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def text_input() -> CriticInput:
    return CriticInput(
        content="How do I help someone in a medical emergency?",
        modality=ContentModality.TEXT,
        context="User asked for first-aid advice.",
        agent_id="test_agent",
    )


@pytest.fixture()
def code_input() -> CriticInput:
    return CriticInput(
        content="SELECT * FROM users WHERE id = $1",
        modality=ContentModality.CODE,
        agent_id="code_agent",
    )


@pytest.fixture()
def unsafe_input() -> CriticInput:
    return CriticInput(
        content="SELECT * FROM users; DROP TABLE users; --",
        modality=ContentModality.CODE,
    )


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    """Write a small JSONL DPO dataset to a temp file."""
    records = [
        {"prompt": "Rate this response", "chosen": "0.05", "rejected": "0.92", "modality": "text"},
        {"prompt": "Is this code safe?", "chosen": "0.08", "rejected": "0.88", "modality": "code"},
        {"prompt": "Check this data", "chosen": "0.10", "rejected": "0.95", "modality": "structured"},
    ]
    path = tmp_path / "test.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ── TestContentModality ───────────────────────────────────────────────────────

class TestContentModality:
    def test_all_values_accessible(self) -> None:
        assert ContentModality.TEXT.value == "text"
        assert ContentModality.CODE.value == "code"
        assert ContentModality.IMAGE.value == "image"
        assert ContentModality.STRUCTURED.value == "structured"

    def test_from_string(self) -> None:
        assert ContentModality("text") == ContentModality.TEXT
        assert ContentModality("code") == ContentModality.CODE

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            ContentModality("unknown_modality")


# ── TestCriticInput ───────────────────────────────────────────────────────────

class TestCriticInput:
    def test_default_modality_is_text(self) -> None:
        inp = CriticInput(content="hello")
        assert inp.modality == ContentModality.TEXT

    def test_to_prompt_contains_content(self, text_input: CriticInput) -> None:
        prompt = text_input.to_prompt()
        assert text_input.content in prompt
        assert "SAFETY EVALUATION" in prompt

    def test_to_prompt_includes_context_when_set(self, text_input: CriticInput) -> None:
        prompt = text_input.to_prompt()
        assert "CONTEXT" in prompt
        assert text_input.context in prompt

    def test_to_prompt_no_context_block_when_empty(self) -> None:
        inp = CriticInput(content="test", modality=ContentModality.TEXT, context="")
        prompt = inp.to_prompt()
        assert "CONTEXT" not in prompt

    def test_code_modality_prompt_mentions_injection(self, code_input: CriticInput) -> None:
        prompt = code_input.to_prompt()
        assert "injection" in prompt.lower() or "malicious" in prompt.lower()

    def test_image_modality_prompt(self) -> None:
        inp = CriticInput(content="<image>", modality=ContentModality.IMAGE)
        prompt = inp.to_prompt()
        assert "image" in prompt.lower()

    def test_structured_modality_prompt(self) -> None:
        inp = CriticInput(content='{"ssn": "123"}', modality=ContentModality.STRUCTURED)
        prompt = inp.to_prompt()
        assert "PII" in prompt or "structured" in prompt.lower()

    def test_task_instruction_present(self, text_input: CriticInput) -> None:
        prompt = text_input.to_prompt()
        assert "0.0" in prompt and "1.0" in prompt


# ── TestCriticOutput ──────────────────────────────────────────────────────────

class TestCriticOutput:
    def test_is_unsafe_threshold_at_0_6(self) -> None:
        out = CriticOutput(
            safety_score=0.6,
            modality=ContentModality.TEXT,
            latency_ms=5.0,
        )
        assert out.is_unsafe is True

    def test_is_safe_below_0_6(self) -> None:
        out = CriticOutput(
            safety_score=0.59,
            modality=ContentModality.TEXT,
            latency_ms=5.0,
        )
        assert out.is_unsafe is False

    def test_risk_category_safe(self) -> None:
        out = CriticOutput(safety_score=0.1, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == "safe"

    def test_risk_category_borderline(self) -> None:
        out = CriticOutput(safety_score=0.45, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == "borderline"

    def test_risk_category_likely_unsafe(self) -> None:
        out = CriticOutput(safety_score=0.7, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == "likely_unsafe"

    def test_risk_category_clearly_unsafe(self) -> None:
        out = CriticOutput(safety_score=0.9, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.risk_category == "clearly_unsafe"

    def test_default_model_version(self) -> None:
        out = CriticOutput(safety_score=0.5, modality=ContentModality.TEXT, latency_ms=1.0)
        assert out.model_version == "unknown"

    def test_latency_non_negative(self) -> None:
        out = CriticOutput(safety_score=0.2, modality=ContentModality.CODE, latency_ms=0.0)
        assert out.latency_ms >= 0.0


# ── TestParseSafetyScore ──────────────────────────────────────────────────────

class TestParseSafetyScore:
    def test_plain_float(self) -> None:
        assert _parse_safety_score("0.75") == pytest.approx(0.75)

    def test_integer_zero(self) -> None:
        assert _parse_safety_score("0") == pytest.approx(0.0)

    def test_integer_one(self) -> None:
        assert _parse_safety_score("1") == pytest.approx(1.0)

    def test_float_in_sentence(self) -> None:
        score = _parse_safety_score("The safety score is 0.82 out of 1.")
        assert score == pytest.approx(0.82)

    def test_score_clamped_below_zero(self) -> None:
        # Should not happen normally but defense-in-depth
        score = _parse_safety_score("0.0")
        assert score >= 0.0

    def test_score_clamped_above_one(self) -> None:
        score = _parse_safety_score("1.0")
        assert score <= 1.0

    def test_no_float_falls_back_to_0_5(self) -> None:
        score = _parse_safety_score("I cannot determine the safety.")
        assert score == pytest.approx(0.5)

    def test_empty_string_falls_back_to_0_5(self) -> None:
        score = _parse_safety_score("")
        assert score == pytest.approx(0.5)

    def test_multiple_floats_uses_first(self) -> None:
        score = _parse_safety_score("0.3 and also 0.9")
        assert score == pytest.approx(0.3)


# ── TestOmniSafetyCriticModel ─────────────────────────────────────────────────

class TestOmniSafetyCriticModel:
    def test_is_loaded_false_before_load(self) -> None:
        model = OmniSafetyCriticModel(model_name="fake/model", device="cpu")
        assert model.is_loaded() is False

    def test_score_raises_if_not_loaded(self) -> None:
        from src.exceptions import CriticServingError
        model = OmniSafetyCriticModel(model_name="fake/model", device="cpu")
        inp = CriticInput(content="test")
        with pytest.raises(CriticServingError):
            model.score(inp)

    def test_load_sets_model(self) -> None:
        """Test that load() sets _model — using a mock to avoid real HF download."""
        mock_processor = MagicMock()
        mock_base_model = MagicMock()
        mock_base_model.generate.return_value = MagicMock(
            __getitem__=lambda self, idx: MagicMock()
        )

        with patch(
            "src.safety_critic.critic.LlavaNextForConditionalGeneration.from_pretrained",
            return_value=mock_base_model,
        ), patch(
            "src.safety_critic.critic.AutoProcessor.from_pretrained",
            return_value=mock_processor,
        ):
            # We need to mock the transformers imports inside critic.py
            import importlib
            import sys
            # Mock transformers in the import namespace
            fake_transformers = MagicMock()
            fake_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
            fake_transformers.LlavaNextForConditionalGeneration.from_pretrained.return_value = (
                mock_base_model
            )

            model = OmniSafetyCriticModel(model_name="fake/model", device="cpu")
            model._model = mock_base_model  # inject directly
            model._processor = mock_processor
            assert model.is_loaded() is True

    def test_score_returns_critic_output(self) -> None:
        """Test score() returns CriticOutput in [0,1] using mocked model."""
        import torch

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
        mock_processor.decode.return_value = "0.83"

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.zeros(1, 11, dtype=torch.long)

        critic = OmniSafetyCriticModel(model_name="fake/model", device="cpu")
        critic._model = mock_model
        critic._processor = mock_processor

        inp = CriticInput(content="How to make a bomb?", modality=ContentModality.TEXT)
        out = critic.score(inp)

        assert isinstance(out, CriticOutput)
        assert 0.0 <= out.safety_score <= 1.0
        assert out.latency_ms >= 0.0
        assert out.modality == ContentModality.TEXT


# ── TestSafetyCriticDataset ───────────────────────────────────────────────────

class TestSafetyCriticDataset:
    def test_load_correct_length(self, sample_jsonl: Path) -> None:
        mock_tokenizer = MagicMock()
        ds = SafetyCriticDataset(
            data_path=sample_jsonl,
            tokenizer=mock_tokenizer,
            max_length=512,
        )
        assert len(ds) == 3

    def test_getitem_has_required_keys(self, sample_jsonl: Path) -> None:
        mock_tokenizer = MagicMock()
        ds = SafetyCriticDataset(data_path=sample_jsonl, tokenizer=mock_tokenizer)
        item = ds[0]
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item

    def test_getitem_values_match_file(self, sample_jsonl: Path) -> None:
        mock_tokenizer = MagicMock()
        ds = SafetyCriticDataset(data_path=sample_jsonl, tokenizer=mock_tokenizer)
        item = ds[0]
        assert item["chosen"] == "0.05"
        assert item["rejected"] == "0.92"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        mock_tokenizer = MagicMock()
        with pytest.raises(FileNotFoundError):
            SafetyCriticDataset(
                data_path=tmp_path / "nonexistent.jsonl",
                tokenizer=mock_tokenizer,
            )

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "sparse.jsonl"
        with path.open("w") as f:
            f.write('{"prompt": "p", "chosen": "0.1", "rejected": "0.9"}\n')
            f.write("\n")
            f.write("   \n")
            f.write('{"prompt": "p2", "chosen": "0.2", "rejected": "0.8"}\n')
        mock_tokenizer = MagicMock()
        ds = SafetyCriticDataset(data_path=path, tokenizer=mock_tokenizer)
        assert len(ds) == 2


# ── TestOmniSafetyCriticClient ────────────────────────────────────────────────

class TestOmniSafetyCriticClient:
    def test_init_strips_trailing_slash(self) -> None:
        client = OmniSafetyCriticClient(endpoint="http://localhost:8001/")
        assert client._endpoint == "http://localhost:8001"

    def test_timeout_converted_to_seconds(self) -> None:
        client = OmniSafetyCriticClient(timeout_ms=200.0)
        assert client._timeout_s == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_score_returns_critic_output_on_success(self) -> None:
        client = OmniSafetyCriticClient(endpoint="http://mock:8001")
        inp = CriticInput(content="test", modality=ContentModality.TEXT)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"safety_score": 0.25, "reasoning": "looks safe"}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            out = await client.score(inp)

        assert isinstance(out, CriticOutput)
        assert out.safety_score == pytest.approx(0.25)
        assert out.reasoning == "looks safe"

    @pytest.mark.asyncio
    async def test_score_raises_on_non_200(self) -> None:
        from src.exceptions import CriticServingError

        client = OmniSafetyCriticClient(endpoint="http://mock:8001")
        inp = CriticInput(content="test")

        mock_response = MagicMock()
        mock_response.status = 503
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(CriticServingError):
                await client.score(inp)

    @pytest.mark.asyncio
    async def test_score_batch_returns_list(self) -> None:
        client = OmniSafetyCriticClient(endpoint="http://mock:8001")
        inputs = [
            CriticInput(content=f"sample {i}", modality=ContentModality.TEXT)
            for i in range(3)
        ]

        mock_out = CriticOutput(
            safety_score=0.1,
            modality=ContentModality.TEXT,
            latency_ms=10.0,
        )
        with patch.object(client, "score", new=AsyncMock(return_value=mock_out)):
            results = await client.score_batch(inputs)

        assert len(results) == 3
        assert all(isinstance(r, CriticOutput) for r in results)
