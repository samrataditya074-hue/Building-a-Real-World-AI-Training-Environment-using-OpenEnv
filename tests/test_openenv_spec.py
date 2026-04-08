"""
tests/test_openenv_spec.py — OpenEnv API specification compliance tests.

Validates that:
  - Pydantic models reject out-of-range values
  - Observation.to_array() returns correct shape
  - reset() / step() / state() follow OpenEnv semantics
"""

import pytest
import numpy as np
from pydantic import ValidationError

from models import Action, Observation
from server.environment import CEOEnvironment


# ─── Pydantic model validation ────────────────────────────────────────────────

class TestActionValidation:
    def test_action_defaults_are_valid(self):
        """Default Action() (all zeros) should be created without error."""
        action = Action()
        assert action.price_adjustment == 0.0
        assert action.marketing_push == 0.0

    def test_action_rejects_price_out_of_range(self):
        """price_adjustment must be in [-1.0, +1.0]."""
        with pytest.raises(ValidationError):
            Action(price_adjustment=1.5)
        with pytest.raises(ValidationError):
            Action(price_adjustment=-1.5)

    def test_action_rejects_negative_marketing(self):
        """marketing_push must be in [0.0, +1.0] — can't spend negative."""
        with pytest.raises(ValidationError):
            Action(marketing_push=-0.1)

    def test_action_rejects_negative_rd(self):
        """rd_investment must be in [0.0, +1.0] — can't invest negative."""
        with pytest.raises(ValidationError):
            Action(rd_investment=-0.5)

    def test_action_accepts_boundary_values(self):
        """Boundary values -1.0 and +1.0 must be accepted."""
        action = Action(
            price_adjustment=-1.0,
            marketing_push=1.0,
            hire_fire=1.0,
            rd_investment=1.0,
            salary_adjustment=-1.0,
            task_allocation=1.0,
            crisis_response=-1.0,
            budget_shift=1.0,
        )
        assert action.price_adjustment == -1.0
        assert action.marketing_push == 1.0


class TestObservationArrayShape:
    def test_to_array_returns_14_floats(self):
        """Observation.to_array() must return shape (14,) float32 array."""
        obs = Observation(
            cash_norm=1.0, revenue_norm=0.5,
            customer_satisfaction_norm=0.75, employee_morale_norm=0.80,
            inventory_norm=0.5, market_trend=1.0, total_employees_norm=0.4,
            brand_reputation_norm=0.6, operational_efficiency_norm=0.7,
            rd_progress_norm=0.0, debt_norm=0.0,
            cash_crisis_flag=0.0, morale_crisis_flag=0.0,
            competitor_price_norm=0.5,
        )
        arr = obs.to_array()
        assert arr.shape == (14,), f"Expected (14,) got {arr.shape}"
        assert arr.dtype == np.float32

    def test_to_array_values_match_fields(self):
        """to_array() index 0 must be cash_norm."""
        obs = Observation(
            cash_norm=0.123, revenue_norm=0.5,
            customer_satisfaction_norm=0.75, employee_morale_norm=0.80,
            inventory_norm=0.5, market_trend=1.0, total_employees_norm=0.4,
            brand_reputation_norm=0.6, operational_efficiency_norm=0.7,
            rd_progress_norm=0.0, debt_norm=0.0,
            cash_crisis_flag=0.0, morale_crisis_flag=0.0,
            competitor_price_norm=0.5,
        )
        arr = obs.to_array()
        assert abs(arr[0] - 0.123) < 1e-5, "Index 0 must be cash_norm"


# ─── Environment API contract ─────────────────────────────────────────────────

class TestEnvironmentContract:
    def setup_method(self):
        self.env = CEOEnvironment()

    def test_reset_returns_observation(self):
        """reset() must return an Observation instance."""
        obs = self.env.reset()
        assert isinstance(obs, Observation)

    def test_reset_done_is_false(self):
        """reset() observation must have done=False."""
        obs = self.env.reset()
        assert obs.done is False

    def test_step_returns_observation(self):
        """step() must return an Observation instance."""
        self.env.reset()
        obs = self.env.step(Action())
        assert isinstance(obs, Observation)

    def test_step_increments_quarter(self):
        """Each step must increment the quarter counter."""
        self.env.reset()
        self.env.step(Action())
        self.env.step(Action())
        assert self.env.typed_state().quarter == 2

    def test_state_returns_dict(self):
        """state() must return a dict (OpenEnv spec)."""
        self.env.reset()
        s = self.env.state()
        assert isinstance(s, dict), "state() must return dict"

    def test_state_has_required_keys(self):
        """state() dict must include canonical OpenEnv keys."""
        self.env.reset()
        s = self.env.state()
        for key in ["quarter", "cash", "profit", "valuation", "employee_morale"]:
            assert key in s, f"Missing key in state(): '{key}'"

    def test_step_info_has_required_fields(self):
        """step() info dict must include pos_reward, neg_reward, thought, actions."""
        self.env.reset()
        obs = self.env.step(Action())
        for field in ["pos_reward", "neg_reward", "thought", "actions", "reward_breakdown"]:
            assert field in obs.info, f"Missing info field: '{field}'"

    def test_reward_breakdown_has_components(self):
        """reward_breakdown must include all shaping components."""
        self.env.reset()
        obs = self.env.step(Action())
        rb = obs.info["reward_breakdown"]
        for component in ["profit_delta", "morale_delta", "rd_payoff", "fire_penalty"]:
            assert component in rb, f"Missing breakdown component: '{component}'"
