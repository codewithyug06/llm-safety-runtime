"""
Unit tests for MOD-04 FederatedRLHF client, server strategy, and privacy wrapper.
All tests use mocks — no real models, no real Flower servers, no real Redis.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.exceptions import FederatedRoundError, PrivacyBudgetExhaustedError
from src.federated_rlhf.client import ArgusFederatedClient
from src.federated_rlhf.privacy import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON_BUDGET,
    DPSGDOpacusWrapper,
    PrivacyAccountingState,
)
from src.federated_rlhf.server import ArgusFedAvgStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def dp_wrapper() -> DPSGDOpacusWrapper:
    return DPSGDOpacusWrapper(epsilon_budget=3.0, delta=1e-5)


@pytest.fixture()
def strategy() -> ArgusFedAvgStrategy:
    return ArgusFedAvgStrategy(
        min_fit_clients=2,
        min_evaluate_clients=2,
        mlflow_experiment="test/federated",
    )


def _make_fit_result(params: list, num_examples: int, metrics: dict) -> MagicMock:
    """Helper to build a mock Flower FitRes."""
    res = MagicMock()
    res.parameters = MagicMock()
    res.num_examples = num_examples
    res.metrics = metrics
    return res


def _make_eval_result(loss: float, num_examples: int, metrics: dict) -> MagicMock:
    """Helper to build a mock Flower EvaluateRes."""
    res = MagicMock()
    res.loss = loss
    res.num_examples = num_examples
    res.metrics = metrics
    return res


# ── TestPrivacyAccountingState ────────────────────────────────────────────────

class TestPrivacyAccountingState:
    def test_initial_epsilon_spent_zero(self) -> None:
        state = PrivacyAccountingState()
        assert state.epsilon_spent == 0.0

    def test_epsilon_remaining_full_at_start(self) -> None:
        state = PrivacyAccountingState(epsilon_budget=3.0)
        assert state.epsilon_remaining == pytest.approx(3.0)

    def test_is_not_exhausted_initially(self) -> None:
        state = PrivacyAccountingState(epsilon_budget=3.0)
        assert state.is_exhausted is False

    def test_is_exhausted_when_spent_equals_budget(self) -> None:
        state = PrivacyAccountingState(epsilon_budget=3.0, epsilon_spent=3.0)
        assert state.is_exhausted is True

    def test_is_exhausted_when_over_budget(self) -> None:
        state = PrivacyAccountingState(epsilon_budget=3.0, epsilon_spent=3.1)
        assert state.is_exhausted is True

    def test_epsilon_remaining_clamped_to_zero(self) -> None:
        state = PrivacyAccountingState(epsilon_budget=3.0, epsilon_spent=4.0)
        assert state.epsilon_remaining == 0.0

    def test_default_delta(self) -> None:
        state = PrivacyAccountingState()
        assert state.delta == DEFAULT_DELTA


# ── TestDPSGDOpacusWrapper ────────────────────────────────────────────────────

class TestDPSGDOpacusWrapper:
    def test_init_defaults(self, dp_wrapper: DPSGDOpacusWrapper) -> None:
        assert dp_wrapper.epsilon_budget == pytest.approx(3.0)
        assert dp_wrapper.delta == pytest.approx(1e-5)
        assert dp_wrapper._attached is False

    def test_check_budget_passes_initially(self, dp_wrapper: DPSGDOpacusWrapper) -> None:
        dp_wrapper.check_budget()  # Should not raise

    def test_check_budget_raises_when_exhausted(self) -> None:
        wrapper = DPSGDOpacusWrapper(epsilon_budget=0.5)
        wrapper._accounting_state.epsilon_spent = 0.5
        with pytest.raises(PrivacyBudgetExhaustedError):
            wrapper.check_budget()

    def test_attach_raises_if_budget_exhausted(self) -> None:
        wrapper = DPSGDOpacusWrapper(epsilon_budget=0.5)
        wrapper._accounting_state.epsilon_spent = 0.5
        with pytest.raises(PrivacyBudgetExhaustedError):
            wrapper.attach(MagicMock(), MagicMock(), MagicMock())

    def test_get_epsilon_raises_if_not_attached(self, dp_wrapper: DPSGDOpacusWrapper) -> None:
        with pytest.raises(RuntimeError):
            dp_wrapper.get_epsilon()

    def test_record_round_raises_if_not_attached(self, dp_wrapper: DPSGDOpacusWrapper) -> None:
        with pytest.raises(RuntimeError):
            dp_wrapper.record_round()

    def test_get_summary_structure(self, dp_wrapper: DPSGDOpacusWrapper) -> None:
        summary = dp_wrapper.get_summary()
        required_keys = {
            "epsilon_spent", "epsilon_budget", "epsilon_remaining",
            "delta", "rounds_completed", "noise_multiplier", "max_grad_norm",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_record_round_raises_at_budget_exhaustion(self) -> None:
        wrapper = DPSGDOpacusWrapper(epsilon_budget=1.0)
        wrapper._attached = True
        mock_engine = MagicMock()
        mock_engine.get_epsilon.return_value = 1.5  # Over budget
        wrapper._privacy_engine = mock_engine

        with pytest.raises(PrivacyBudgetExhaustedError):
            wrapper.record_round()

    def test_record_round_updates_state(self) -> None:
        wrapper = DPSGDOpacusWrapper(epsilon_budget=3.0)
        wrapper._attached = True
        mock_engine = MagicMock()
        mock_engine.get_epsilon.return_value = 0.3
        wrapper._privacy_engine = mock_engine

        delta_eps = wrapper.record_round()
        assert delta_eps == pytest.approx(0.3)
        assert wrapper.accounting_state.epsilon_spent == pytest.approx(0.3)
        assert wrapper.accounting_state.rounds_completed == 1

    def test_privacy_budget_error_attributes(self) -> None:
        exc = PrivacyBudgetExhaustedError(current_epsilon=3.1)
        assert "3.1" in str(exc)

    def test_attach_raises_import_error_when_opacus_missing(self) -> None:
        wrapper = DPSGDOpacusWrapper()
        with patch.dict("sys.modules", {"opacus": None}):
            with pytest.raises((ImportError, Exception)):
                wrapper.attach(MagicMock(), MagicMock(), MagicMock())


# ── TestArgusFedAvgStrategy ───────────────────────────────────────────────────

class TestArgusFedAvgStrategy:
    def _make_params(self, shapes=None) -> list:
        """Create a list of random numpy parameter arrays."""
        if shapes is None:
            shapes = [(10, 5), (5,), (5, 3), (3,)]
        return [np.random.randn(*s).astype(np.float32) for s in shapes]

    def _mock_fit_result(self, params, num_examples, metrics=None):
        """Create a mock FitRes with parameters_to_ndarrays patched."""
        res = MagicMock()
        res.num_examples = num_examples
        res.metrics = metrics or {}
        res.parameters = MagicMock()
        return res, params

    def test_weighted_average_equal_weights(self, strategy: ArgusFedAvgStrategy) -> None:
        params_a = [np.array([1.0, 2.0], dtype=np.float32)]
        params_b = [np.array([3.0, 4.0], dtype=np.float32)]
        averaged = strategy._weighted_average([(params_a, 100), (params_b, 100)])
        expected = np.array([2.0, 3.0])
        np.testing.assert_allclose(averaged[0], expected, atol=1e-5)

    def test_weighted_average_unequal_weights(self, strategy: ArgusFedAvgStrategy) -> None:
        params_a = [np.array([0.0], dtype=np.float32)]
        params_b = [np.array([10.0], dtype=np.float32)]
        # 3:1 ratio → result should be 7.5
        averaged = strategy._weighted_average([(params_a, 3), (params_b, 1)])
        np.testing.assert_allclose(averaged[0], np.array([2.5]), atol=1e-5)

    def test_weighted_average_raises_on_zero_total(self, strategy: ArgusFedAvgStrategy) -> None:
        from src.exceptions import GradientAggregationError
        params = [np.array([1.0])]
        with pytest.raises(GradientAggregationError):
            strategy._weighted_average([(params, 0), (params, 0)])

    def test_aggregate_fit_raises_on_empty_results(self, strategy: ArgusFedAvgStrategy) -> None:
        from src.exceptions import GradientAggregationError
        with pytest.raises(GradientAggregationError):
            strategy.aggregate_fit(server_round=1, results=[], failures=[])

    def test_aggregate_fit_raises_insufficient_clients(self, strategy: ArgusFedAvgStrategy) -> None:
        from src.exceptions import FederatedRoundError
        # Strategy requires min 2 clients; give only 1
        params = self._make_params()
        res = MagicMock()
        res.num_examples = 100
        res.metrics = {}
        res.parameters = MagicMock()

        with patch.object(
            strategy, "_parameters_to_ndarrays", return_value=params
        ), patch.object(strategy, "_register_to_mlflow"):
            with pytest.raises(FederatedRoundError):
                strategy.aggregate_fit(server_round=1, results=[(None, res)], failures=[])

    def test_aggregate_fit_success_with_two_clients(self, strategy: ArgusFedAvgStrategy) -> None:
        params_a = self._make_params()
        params_b = self._make_params()

        res_a = MagicMock()
        res_a.num_examples = 200
        res_a.metrics = {"train_loss": 0.4, "epsilon_spent": 0.2}

        res_b = MagicMock()
        res_b.num_examples = 300
        res_b.metrics = {"train_loss": 0.3, "epsilon_spent": 0.25}

        with patch.object(
            strategy, "_parameters_to_ndarrays", side_effect=[params_a, params_b]
        ), patch.object(strategy, "_register_to_mlflow"):
            aggregated, metrics = strategy.aggregate_fit(
                server_round=1,
                results=[(None, res_a), (None, res_b)],
                failures=[],
            )

        assert aggregated is not None
        assert len(aggregated) == len(params_a)
        assert "avg_train_loss" in metrics
        assert metrics["num_clients"] == 2
        assert metrics["total_examples"] == 500

    def test_aggregate_evaluate_weighted_loss(self, strategy: ArgusFedAvgStrategy) -> None:
        res_a = MagicMock()
        res_a.loss = 0.4
        res_a.num_examples = 100
        res_a.metrics = {"safety_accuracy": 0.8}

        res_b = MagicMock()
        res_b.loss = 0.2
        res_b.num_examples = 100
        res_b.metrics = {"safety_accuracy": 0.9}

        loss, metrics = strategy.aggregate_evaluate(
            server_round=1,
            results=[(None, res_a), (None, res_b)],
            failures=[],
        )
        # Weighted loss: (0.4*100 + 0.2*100) / 200 = 0.3
        assert loss == pytest.approx(0.3)
        assert metrics["safety_accuracy"] == pytest.approx(0.85)

    def test_aggregate_evaluate_no_results_returns_none(self, strategy: ArgusFedAvgStrategy) -> None:
        loss, metrics = strategy.aggregate_evaluate(
            server_round=1, results=[], failures=[]
        )
        assert loss is None
        assert metrics == {}


# ── TestArgusFederatedClient ──────────────────────────────────────────────────

class TestArgusFederatedClient:
    """Tests for ArgusFederatedClient — all heavy deps (torch, peft) are mocked."""

    @pytest.fixture()
    def mock_dp(self) -> MagicMock:
        dp = MagicMock(spec=DPSGDOpacusWrapper)
        state = PrivacyAccountingState(epsilon_budget=3.0, epsilon_spent=0.3)
        dp.accounting_state = state
        dp.check_budget.return_value = None
        dp.record_round.return_value = 0.3
        return dp

    @pytest.fixture()
    def client(self, mock_dp: MagicMock) -> ArgusFederatedClient:
        return ArgusFederatedClient(
            client_id="test_node",
            model_name="dummy/model",
            train_data_path="data/safety_critic/train.jsonl",
            val_data_path="data/safety_critic/val.jsonl",
            dp_wrapper=mock_dp,
            device="cpu",
        )

    def _fake_lora_params(self) -> list:
        return [np.ones((4, 4), dtype=np.float32), np.zeros((4,), dtype=np.float32)]

    # ── __init__ tests ─────────────────────────────────────────────────────────

    def test_init_stores_client_id(self, client: ArgusFederatedClient) -> None:
        assert client.client_id == "test_node"

    def test_init_model_is_none(self, client: ArgusFederatedClient) -> None:
        assert client._model is None

    def test_init_round_zero(self, client: ArgusFederatedClient) -> None:
        assert client._round == 0

    def test_init_creates_default_dp_wrapper(self) -> None:
        c = ArgusFederatedClient(client_id="x")
        assert isinstance(c._dp_wrapper, DPSGDOpacusWrapper)

    def test_init_uses_provided_dp_wrapper(self, client: ArgusFederatedClient, mock_dp: MagicMock) -> None:
        assert client._dp_wrapper is mock_dp

    # ── get_parameters tests ───────────────────────────────────────────────────

    def test_get_parameters_loads_model_when_none(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        with patch.object(client, "_load_model") as mock_load, \
             patch("src.federated_rlhf.client._get_lora_params", return_value=params):
            client._model = None
            result = client.get_parameters(config={})
            mock_load.assert_called_once()
            assert result == params

    def test_get_parameters_skips_load_when_model_set(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch.object(client, "_load_model") as mock_load, \
             patch("src.federated_rlhf.client._get_lora_params", return_value=params):
            result = client.get_parameters(config={})
            mock_load.assert_not_called()
            assert result == params

    def test_get_parameters_returns_list_of_ndarrays(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._get_lora_params", return_value=params):
            result = client.get_parameters(config={})
        assert all(isinstance(p, np.ndarray) for p in result)

    # ── fit tests ─────────────────────────────────────────────────────────────

    def test_fit_checks_privacy_budget(
        self, client: ArgusFederatedClient, mock_dp: MagicMock
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch("src.federated_rlhf.client._get_lora_params", return_value=params), \
             patch.object(client, "_run_local_training", return_value=(0.4, 100)):
            client.fit(parameters=params, config={"round": 1})
        mock_dp.check_budget.assert_called_once()

    def test_fit_returns_params_and_metrics(
        self, client: ArgusFederatedClient, mock_dp: MagicMock
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch("src.federated_rlhf.client._get_lora_params", return_value=params), \
             patch.object(client, "_run_local_training", return_value=(0.35, 120)):
            result_params, num_examples, metrics = client.fit(
                parameters=params, config={"round": 2}
            )
        assert result_params == params
        assert num_examples == 120
        assert "train_loss" in metrics
        assert metrics["round"] == 2
        assert "epsilon_spent" in metrics

    def test_fit_re_raises_privacy_budget_error(
        self, client: ArgusFederatedClient, mock_dp: MagicMock
    ) -> None:
        mock_dp.check_budget.side_effect = PrivacyBudgetExhaustedError(current_epsilon=3.1)
        params = self._fake_lora_params()
        with pytest.raises(PrivacyBudgetExhaustedError):
            client.fit(parameters=params, config={"round": 1})

    def test_fit_wraps_training_exception_as_federated_round_error(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch.object(client, "_run_local_training", side_effect=RuntimeError("GPU OOM")):
            with pytest.raises(FederatedRoundError) as exc_info:
                client.fit(parameters=params, config={"round": 1})
        assert "GPU OOM" in str(exc_info.value)

    def test_fit_increments_round_from_config(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch("src.federated_rlhf.client._get_lora_params", return_value=params), \
             patch.object(client, "_run_local_training", return_value=(0.3, 50)):
            client.fit(parameters=params, config={"round": 7})
        assert client._round == 7

    def test_fit_calls_record_round(
        self, client: ArgusFederatedClient, mock_dp: MagicMock
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch("src.federated_rlhf.client._get_lora_params", return_value=params), \
             patch.object(client, "_run_local_training", return_value=(0.4, 100)):
            client.fit(parameters=params, config={"round": 1})
        mock_dp.record_round.assert_called_once()

    # ── evaluate tests ────────────────────────────────────────────────────────

    def test_evaluate_returns_loss_and_metrics(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch.object(client, "_run_local_eval", return_value=(0.25, 0.88, 80)):
            loss, num_examples, metrics = client.evaluate(
                parameters=params, config={}
            )
        assert loss == pytest.approx(0.25)
        assert num_examples == 80
        assert metrics["safety_accuracy"] == pytest.approx(0.88)
        assert metrics["client_id"] == "test_node"

    def test_evaluate_loads_model_when_none(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = None
        with patch.object(client, "_load_model") as mock_load, \
             patch("src.federated_rlhf.client._set_lora_params"), \
             patch.object(client, "_run_local_eval", return_value=(0.3, 0.85, 60)):
            client.evaluate(parameters=params, config={})
        mock_load.assert_called_once()

    def test_evaluate_val_loss_in_metrics(
        self, client: ArgusFederatedClient
    ) -> None:
        params = self._fake_lora_params()
        client._model = MagicMock()
        with patch("src.federated_rlhf.client._set_lora_params"), \
             patch.object(client, "_run_local_eval", return_value=(0.18, 0.91, 45)):
            _, _, metrics = client.evaluate(parameters=params, config={})
        assert "val_loss" in metrics
        assert metrics["val_loss"] == pytest.approx(0.18)
