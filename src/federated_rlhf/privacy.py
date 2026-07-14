from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import structlog

from src.exceptions import PrivacyBudgetExhaustedError

logger = structlog.get_logger(__name__)

DEFAULT_EPSILON_BUDGET: float = 3.0
DEFAULT_DELTA: float = 1e-5
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_NOISE_MULTIPLIER: float = 1.1


@dataclass
class PrivacyAccountingState:
    epsilon_budget: float = DEFAULT_EPSILON_BUDGET
    delta: float = DEFAULT_DELTA
    epsilon_spent: float = 0.0
    rounds_completed: int = 0
    epsilon_per_round: list = field(default_factory=list)

    @property
    def epsilon_remaining(self) -> float:
        return max(0.0, self.epsilon_budget - self.epsilon_spent)

    @property
    def is_exhausted(self) -> bool:
        return self.epsilon_spent >= self.epsilon_budget


class DPSGDOpacusWrapper:
    def __init__(
        self,
        epsilon_budget: float = DEFAULT_EPSILON_BUDGET,
        delta: float = DEFAULT_DELTA,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
    ) -> None:
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

        self._privacy_engine: Optional[Any] = None
        self._accounting_state = PrivacyAccountingState(
            epsilon_budget=epsilon_budget,
            delta=delta,
        )
        self._attached = False

    def attach(
        self,
        model: Any,
        optimizer: Any,
        data_loader: Any,
    ) -> Tuple[Any, Any, Any]:

        if self._accounting_state.is_exhausted:
            raise PrivacyBudgetExhaustedError(
                current_epsilon=self._accounting_state.epsilon_spent,
            )

        try:
            from opacus import PrivacyEngine
        except ImportError:
            raise ImportError("Run: pip install opacus>=1.4.0")

        self._privacy_engine = PrivacyEngine()
        private_model, private_optimizer, private_loader = (
            self._privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        )
        self._attached = True

        logger.info(
            "dp_sgd_attached",
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            delta=self.delta,
        )
        return private_model, private_optimizer, private_loader

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        if not self._attached or self._privacy_engine is None:
            raise RuntimeError("PrivacyEngine not attached — call attach() first")

        delta = delta or self.delta
        epsilon = self._privacy_engine.get_epsilon(delta=delta)
        return float(epsilon)

    def record_round(self) -> float:
        current_epsilon = self.get_epsilon()
        prev_epsilon = self._accounting_state.epsilon_spent

        delta_epsilon = max(0.0, current_epsilon - prev_epsilon)
        self._accounting_state.epsilon_spent = current_epsilon
        self._accounting_state.rounds_completed += 1
        self._accounting_state.epsilon_per_round.append(delta_epsilon)

        logger.info(
            "privacy_round_recorded",
            round=self._accounting_state.rounds_completed,
            epsilon_this_round=f"{delta_epsilon:.4f}",
            epsilon_total=f"{current_epsilon:.4f}",
            epsilon_budget=self.epsilon_budget,
            epsilon_remaining=f"{self._accounting_state.epsilon_remaining:.4f}",
        )

        if self._accounting_state.is_exhausted:
            raise PrivacyBudgetExhaustedError(
                current_epsilon=current_epsilon,
            )

        return delta_epsilon

    def check_budget(self) -> None:
        if self._accounting_state.is_exhausted:
            raise PrivacyBudgetExhaustedError(
                current_epsilon=self._accounting_state.epsilon_spent,
            )

    @property
    def accounting_state(self) -> PrivacyAccountingState:
        return self._accounting_state

    def get_summary(self) -> dict:

        state = self._accounting_state
        return {
            "epsilon_spent": state.epsilon_spent,
            "epsilon_budget": state.epsilon_budget,
            "epsilon_remaining": state.epsilon_remaining,
            "delta": self.delta,
            "rounds_completed": state.rounds_completed,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
        }
