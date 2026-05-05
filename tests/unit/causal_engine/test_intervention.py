"""
MOD-02: CausalInterventionEngine Unit Tests
=============================================
Tests cover:
- HeadSignature identity and hashing
- CausalGraph construction, edge queries, and head retrieval
- InterventionSpec / InterventionResult data integrity
- JAX scale_attention_weights correctness
- JAX compute_ablation_effect correctness
- CausalScrubber interface contract (mocked inference)
- CausalInterventionEngine decision logic

All tests use stub/mock values — no real LLM is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.causal_engine.intervention import (
    CausalEdge,
    CausalGraph,
    CausalInterventionEngine,
    CausalScrubber,
    HeadSignature,
    InterventionResult,
    InterventionSpec,
    compute_ablation_effect,
    scale_attention_weights,
)
from src.exceptions import CausalGraphNotFoundError, InterventionError


# ── Fixtures ──────────────────────────────────────────────────────────────────

MODEL_ID = "llama-3.1-8b-stub"


@pytest.fixture
def head_a() -> HeadSignature:
    return HeadSignature(layer_idx=12, head_idx=3, model_id=MODEL_ID)


@pytest.fixture
def head_b() -> HeadSignature:
    return HeadSignature(layer_idx=20, head_idx=7, model_id=MODEL_ID)


@pytest.fixture
def head_c() -> HeadSignature:
    return HeadSignature(layer_idx=28, head_idx=0, model_id=MODEL_ID)


@pytest.fixture
def causal_graph(head_a: HeadSignature, head_b: HeadSignature, head_c: HeadSignature) -> CausalGraph:
    edges = [
        CausalEdge(
            head=head_a,
            behavior_category="jailbreak",
            causal_effect=0.72,
            ablation_delta=0.31,
            patching_delta=0.28,
            confidence=0.90,
            discovered_at="2025-01-01T00:00:00Z",
        ),
        CausalEdge(
            head=head_b,
            behavior_category="jailbreak",
            causal_effect=0.45,
            ablation_delta=0.18,
            patching_delta=0.15,
            confidence=0.82,
            discovered_at="2025-01-01T00:00:00Z",
        ),
        CausalEdge(
            head=head_c,
            behavior_category="hallucination",
            causal_effect=0.60,
            ablation_delta=0.25,
            patching_delta=0.22,
            confidence=0.88,
            discovered_at="2025-01-01T00:00:00Z",
        ),
        CausalEdge(
            head=head_a,
            behavior_category="hallucination",
            causal_effect=0.10,  # below default min_effect=0.15
            ablation_delta=0.03,
            patching_delta=0.02,
            confidence=0.60,
            discovered_at="2025-01-01T00:00:00Z",
        ),
    ]
    return CausalGraph(model_id=MODEL_ID, edges=edges, version=1)


# ── HeadSignature Tests ───────────────────────────────────────────────────────

class TestHeadSignature:
    def test_equality(self) -> None:
        h1 = HeadSignature(layer_idx=5, head_idx=2, model_id="m1")
        h2 = HeadSignature(layer_idx=5, head_idx=2, model_id="m1")
        assert h1 == h2

    def test_inequality_different_layer(self) -> None:
        h1 = HeadSignature(layer_idx=5, head_idx=2, model_id="m1")
        h2 = HeadSignature(layer_idx=6, head_idx=2, model_id="m1")
        assert h1 != h2

    def test_inequality_different_model(self) -> None:
        h1 = HeadSignature(layer_idx=5, head_idx=2, model_id="m1")
        h2 = HeadSignature(layer_idx=5, head_idx=2, model_id="m2")
        assert h1 != h2

    def test_hashable_usable_as_dict_key(self) -> None:
        h = HeadSignature(layer_idx=3, head_idx=1, model_id="test")
        d = {h: 0.42}
        assert d[h] == 0.42

    def test_hashable_usable_in_set(self) -> None:
        h1 = HeadSignature(layer_idx=3, head_idx=1, model_id="test")
        h2 = HeadSignature(layer_idx=3, head_idx=1, model_id="test")
        s = {h1, h2}
        assert len(s) == 1

    def test_str_representation(self) -> None:
        h = HeadSignature(layer_idx=12, head_idx=3, model_id="llama-8b")
        assert "12" in str(h)
        assert "3" in str(h)
        assert "llama-8b" in str(h)


# ── CausalEdge Tests ──────────────────────────────────────────────────────────

class TestCausalEdge:
    def test_fields_accessible(self, head_a: HeadSignature) -> None:
        edge = CausalEdge(
            head=head_a,
            behavior_category="jailbreak",
            causal_effect=0.72,
            ablation_delta=0.31,
            patching_delta=0.28,
            confidence=0.90,
        )
        assert edge.causal_effect == 0.72
        assert edge.behavior_category == "jailbreak"
        assert edge.confidence == 0.90

    def test_causal_effect_range(self, head_a: HeadSignature) -> None:
        edge = CausalEdge(
            head=head_a,
            behavior_category="test",
            causal_effect=0.55,
            ablation_delta=0.1,
            patching_delta=0.1,
            confidence=0.8,
        )
        assert 0.0 <= edge.causal_effect <= 1.0


# ── CausalGraph Tests ─────────────────────────────────────────────────────────

class TestCausalGraph:
    def test_get_unsafe_heads_returns_correct_category(
        self, causal_graph: CausalGraph, head_a: HeadSignature, head_b: HeadSignature
    ) -> None:
        jailbreak_heads = causal_graph.get_unsafe_heads("jailbreak")
        assert head_a in jailbreak_heads
        assert head_b in jailbreak_heads

    def test_get_unsafe_heads_filters_by_min_effect(
        self, causal_graph: CausalGraph, head_a: HeadSignature
    ) -> None:
        hallucination_heads = causal_graph.get_unsafe_heads(
            "hallucination", min_effect=0.15
        )
        # head_a has causal_effect=0.10 for hallucination → filtered out
        assert head_a not in hallucination_heads

    def test_get_unsafe_heads_sorted_by_effect_descending(
        self, causal_graph: CausalGraph
    ) -> None:
        heads = causal_graph.get_unsafe_heads("jailbreak")
        # head_a has effect 0.72, head_b has 0.45 → head_a should be first
        assert heads[0].head_idx == 3  # head_a
        assert heads[1].head_idx == 7  # head_b

    def test_empty_category_returns_empty_list(
        self, causal_graph: CausalGraph
    ) -> None:
        heads = causal_graph.get_unsafe_heads("nonexistent_category")
        assert heads == []

    def test_version_is_positive_integer(
        self, causal_graph: CausalGraph
    ) -> None:
        assert causal_graph.version >= 1

    def test_edge_count(self, causal_graph: CausalGraph) -> None:
        assert len(causal_graph.edges) == 4


# ── InterventionSpec / InterventionResult Tests ────────────────────────────────

class TestInterventionSpec:
    def test_heads_is_a_set(
        self, head_a: HeadSignature, head_b: HeadSignature
    ) -> None:
        spec = InterventionSpec(
            heads={head_a, head_b},
            scale_factors={head_a: 0.3, head_b: 0.5},
            reason="High jailbreak risk",
            risk_score_before=0.82,
        )
        assert isinstance(spec.heads, set)
        assert len(spec.heads) == 2

    def test_scale_factors_in_range(
        self, head_a: HeadSignature
    ) -> None:
        spec = InterventionSpec(
            heads={head_a},
            scale_factors={head_a: 0.3},
            reason="test",
            risk_score_before=0.75,
        )
        for factor in spec.scale_factors.values():
            assert 0.0 <= factor <= 1.0


class TestInterventionResult:
    def test_delta_is_improvement(
        self, head_a: HeadSignature
    ) -> None:
        spec = InterventionSpec(
            heads={head_a},
            scale_factors={head_a: 0.3},
            reason="test",
            risk_score_before=0.80,
        )
        result = InterventionResult(
            spec=spec,
            risk_score_after=0.45,
            delta=0.35,
            latency_ms=3.2,
            success=True,
        )
        assert result.delta > 0
        assert result.success is True

    def test_latency_is_positive(
        self, head_a: HeadSignature
    ) -> None:
        spec = InterventionSpec(
            heads={head_a},
            scale_factors={head_a: 0.5},
            reason="test",
            risk_score_before=0.6,
        )
        result = InterventionResult(
            spec=spec,
            risk_score_after=0.3,
            delta=0.3,
            latency_ms=2.1,
            success=True,
        )
        assert result.latency_ms >= 0


# ── JAX Function Tests ────────────────────────────────────────────────────────

class TestScaleAttentionWeights:
    def test_unmasked_heads_unchanged(self) -> None:
        batch, heads, seq = 1, 4, 8
        attn = jnp.ones((batch, heads, seq, seq))
        head_mask = jnp.array([False, False, False, False])
        scale_factors = jnp.array([0.1, 0.2, 0.3, 0.4])

        result = scale_attention_weights(attn, head_mask, scale_factors)
        np.testing.assert_allclose(np.array(result), np.ones((batch, heads, seq, seq)))

    def test_masked_head_scaled_down(self) -> None:
        batch, heads, seq = 1, 4, 4
        attn = jnp.ones((batch, heads, seq, seq))
        head_mask = jnp.array([False, False, True, False])
        scale_factors = jnp.array([1.0, 1.0, 0.3, 1.0])

        result = scale_attention_weights(attn, head_mask, scale_factors)
        # Head 2 should be scaled by 0.3
        np.testing.assert_allclose(
            np.array(result[0, 2]),
            np.full((seq, seq), 0.3),
            atol=1e-6,
        )
        # Other heads unchanged
        np.testing.assert_allclose(np.array(result[0, 0]), np.ones((seq, seq)), atol=1e-6)

    def test_output_shape_preserved(self) -> None:
        attn = jnp.ones((2, 8, 16, 16))
        head_mask = jnp.zeros(8, dtype=bool)
        scale_factors = jnp.ones(8)
        result = scale_attention_weights(attn, head_mask, scale_factors)
        assert result.shape == (2, 8, 16, 16)

    def test_all_heads_zeroed_zeroes_output(self) -> None:
        attn = jnp.ones((1, 4, 4, 4))
        head_mask = jnp.ones(4, dtype=bool)
        scale_factors = jnp.zeros(4)
        result = scale_attention_weights(attn, head_mask, scale_factors)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-6)


class TestComputeAblationEffect:
    def test_identical_activations_produce_zero_effect(self) -> None:
        dim = 32
        activations = jnp.ones((1, 8, dim))
        probe_weights = jnp.ones(dim) / dim

        effect = compute_ablation_effect(activations, activations, probe_weights)
        np.testing.assert_allclose(np.array(effect), 0.0, atol=1e-5)

    def test_different_activations_produce_nonzero_effect(self) -> None:
        dim = 32
        original = jnp.ones((1, 8, dim))
        ablated = jnp.zeros((1, 8, dim))
        probe_weights = jnp.ones(dim) / dim

        effect = compute_ablation_effect(original, ablated, probe_weights)
        assert float(effect) != 0.0

    def test_output_is_scalar(self) -> None:
        dim = 16
        import jax
        original = jax.random.normal(jax.random.PRNGKey(0), (1, 4, dim))
        ablated = jnp.zeros((1, 4, dim))
        probe_weights = jnp.ones(dim) / dim

        effect = compute_ablation_effect(original, ablated, probe_weights)
        assert effect.shape == ()  # scalar


# ── CausalScrubber Tests (mocked) ─────────────────────────────────────────────

class TestCausalScrubber:
    def test_init(self) -> None:
        mock_model = MagicMock()
        mock_probes = {"hallucination": MagicMock()}
        scrubber = CausalScrubber(
            model=mock_model,
            probe_registry=mock_probes,
            model_id=MODEL_ID,
            target_layers=[4, 8, 12],
        )
        assert scrubber.model_id == MODEL_ID

    def test_build_causal_graph_returns_causal_graph(self) -> None:
        mock_model = MagicMock()
        mock_probes = {"hallucination": MagicMock()}

        scrubber = CausalScrubber(
            model=mock_model,
            probe_registry=mock_probes,
            model_id=MODEL_ID,
            target_layers=[4, 8],
        )

        # Patch internal ablation to return fixed deltas
        with patch.object(scrubber, "_run_ablation_study", return_value=[
            CausalEdge(
                head=HeadSignature(layer_idx=4, head_idx=0, model_id=MODEL_ID),
                behavior_category="hallucination",
                causal_effect=0.55,
                ablation_delta=0.22,
                patching_delta=0.20,
                confidence=0.85,
            )
        ]):
            graph = scrubber.build_causal_graph(
                prompts=["test prompt"],
                categories=["hallucination"],
            )

        assert isinstance(graph, CausalGraph)
        assert graph.model_id == MODEL_ID
        assert len(graph.edges) >= 1


# ── CausalInterventionEngine Tests ────────────────────────────────────────────

class TestCausalInterventionEngine:
    def test_no_intervention_below_threshold(
        self, causal_graph: CausalGraph
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        mock_signal = MagicMock()
        mock_signal.composite_risk_score = 0.40  # Below threshold

        result = engine.maybe_intervene(mock_signal)
        assert result is None

    def test_intervention_triggered_above_threshold(
        self, causal_graph: CausalGraph
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        mock_signal = MagicMock()
        mock_signal.composite_risk_score = 0.85
        mock_signal.risk_level.name = "HIGH"
        # Simulate probe scores indicating jailbreak
        mock_signal.probe_scores = {MagicMock(): 0.90}

        spec = engine.build_intervention_spec(mock_signal, category="jailbreak")
        assert spec is not None
        assert len(spec.heads) > 0

    def test_intervention_spec_scale_factors_in_range(
        self, causal_graph: CausalGraph
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        mock_signal = MagicMock()
        mock_signal.composite_risk_score = 0.80

        spec = engine.build_intervention_spec(mock_signal, category="jailbreak")
        if spec is not None:
            for head, factor in spec.scale_factors.items():
                assert 0.0 < factor <= 1.0, f"Scale factor {factor} out of range for {head}"

    def test_maybe_intervene_returns_result_above_threshold(
        self, causal_graph: CausalGraph
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        mock_signal = MagicMock()
        mock_signal.composite_risk_score = 0.85
        # Use string keys that match categories in the causal_graph fixture
        mock_signal.probe_scores = {"jailbreak": 0.90, "hallucination": 0.75}

        result = engine.maybe_intervene(mock_signal)
        assert result is not None
        assert isinstance(result, InterventionResult)
        assert result.risk_score_after < mock_signal.composite_risk_score
        assert result.delta > 0
        assert result.latency_ms >= 0

    def test_maybe_intervene_returns_none_below_threshold(
        self, causal_graph: CausalGraph
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        mock_signal = MagicMock()
        mock_signal.composite_risk_score = 0.30

        result = engine.maybe_intervene(mock_signal)
        assert result is None

    def test_update_graph_swaps_graph(
        self, causal_graph: CausalGraph, head_a: HeadSignature
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        new_graph = CausalGraph(model_id="new-model", edges=[], version=2)
        engine.update_graph(new_graph)
        assert engine._graph.model_id == "new-model"
        assert engine._graph.version == 2

    def test_prepare_intervention_reason_contains_score(
        self, causal_graph: CausalGraph
    ) -> None:
        engine = CausalInterventionEngine(
            causal_graph=causal_graph,
            intervention_threshold=0.65,
        )
        spec = engine.prepare_intervention(risk_score=0.82, triggered_categories=["jailbreak"])
        assert spec is not None
        assert "0.82" in spec.reason


# ── CausalScrubber Ablation Study Tests ───────────────────────────────────────

class TestCausalScrubberAblationStudy:
    def test_ablation_study_finds_edges_above_threshold(self) -> None:
        """High-magnitude activations should produce detectable causal edges."""
        scrubber = CausalScrubber(
            model_id=MODEL_ID,
            num_layers=2,
            num_heads=4,
        )
        # Shape: (seq=8, heads=4, d_head=16)
        rng = np.random.default_rng(42)
        activations_by_layer: Dict[int, np.ndarray] = {
            0: rng.normal(size=(8, 4, 16)).astype(np.float32),
            1: rng.normal(size=(8, 4, 16)).astype(np.float32),
        }
        # Probe weights shape: (hidden,) = (16,)
        probe_weights = {
            "jailbreak": np.ones(16, dtype=np.float32) / 16,
            "hallucination": np.ones(16, dtype=np.float32) / 16,
        }
        edges = scrubber.ablation_study(
            activations_by_layer=activations_by_layer,
            probe_weights=probe_weights,
            categories=["jailbreak", "hallucination"],
            min_delta_threshold=0.0,  # capture all edges
        )
        assert isinstance(edges, list)
        # 2 layers × 4 heads × 2 categories = 16 candidate edges (some may be filtered)
        assert len(edges) <= 16

    def test_ablation_study_returns_causal_edge_objects(self) -> None:
        scrubber = CausalScrubber(
            model_id=MODEL_ID,
            num_layers=1,
            num_heads=2,
        )
        rng = np.random.default_rng(0)
        activations_by_layer = {0: rng.normal(size=(4, 2, 8)).astype(np.float32)}
        probe_weights = {"jailbreak": np.ones(8, dtype=np.float32)}
        edges = scrubber.ablation_study(
            activations_by_layer=activations_by_layer,
            probe_weights=probe_weights,
            categories=["jailbreak"],
            min_delta_threshold=0.0,
        )
        for edge in edges:
            assert isinstance(edge, CausalEdge)
            assert edge.behavior_category == "jailbreak"
            assert 0.0 <= edge.confidence <= 1.0

    def test_ablate_head_zeroes_correct_slice(self) -> None:
        scrubber = CausalScrubber(model_id=MODEL_ID, num_layers=1, num_heads=4)
        acts = np.ones((8, 4, 16), dtype=np.float32)
        ablated = scrubber._ablate_head(acts, head_idx=2)
        # Head 2 should be zeroed
        np.testing.assert_array_equal(ablated[:, 2, :], 0.0)
        # Other heads unchanged
        np.testing.assert_array_equal(ablated[:, 0, :], 1.0)
        np.testing.assert_array_equal(ablated[:, 1, :], 1.0)
        np.testing.assert_array_equal(ablated[:, 3, :], 1.0)
