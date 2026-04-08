"""
tests/test_ceo_env.py — Core environment integration tests.
"""

import pytest
import numpy as np

from models import Action, Observation, State
from server.environment import CEOEnvironment
from tasks import grade_easy_survival, grade_medium_profit, grade_hard_utopia


def test_environment_reset():
    """reset() should return a valid Observation and set the state to starting values."""
    env = CEOEnvironment()
    obs = env.reset()

    assert isinstance(obs, Observation)
    assert 0.0 <= obs.cash_norm <= 2.0

    # typed_state() gives access to the full State object for assertions
    s = env.typed_state()
    assert s.cash == 200_000.0, (
        "Starting cash should be $200,000 (see models.py State.cash default)"
    )
    assert s.total_employees == 20
    assert s.quarter == 0


def test_environment_reset_with_seed():
    """Two resets with the same seed should produce the same employee roster."""
    env = CEOEnvironment()
    obs_a = env.reset(seed=42)
    emp_names_a = [e.name for e in env.typed_state().employees]

    obs_b = env.reset(seed=42)
    emp_names_b = [e.name for e in env.typed_state().employees]

    assert emp_names_a == emp_names_b, "Same seed must produce same initial employee roster"


def test_environment_step():
    """step() should update state and return a valid Observation."""
    env = CEOEnvironment()
    env.reset()
    action = Action(price_adjustment=0.1, hire_fire=0.5)  # hire some people

    result = env.step(action)

    assert isinstance(result, Observation)
    assert not result.done
    assert "pos_reward" in result.info
    assert "thought" in result.info
    assert "reward_breakdown" in result.info

    s = env.typed_state()
    assert s.total_employees > 20, "Hiring action should increase employee count"
    assert s.quarter == 1


def test_environment_state_returns_dict():
    """state() must return a dict (OpenEnv spec requirement)."""
    env = CEOEnvironment()
    env.reset()
    s = env.state()

    assert isinstance(s, dict), "state() must return dict per OpenEnv spec"

    required_keys = [
        "quarter", "cash", "revenue", "profit",
        "employee_morale", "customer_satisfaction",
        "total_employees", "valuation", "dept_scores",
    ]
    for key in required_keys:
        assert key in s, f"state() dict missing required key: '{key}'"


def test_graders():
    """Verify all three final-state graders return correct values."""
    state = State(
        cash=1_000.0,
        total_employees=5,
        customer_satisfaction=50.0,
        profit=5_000.0,
        employee_morale=95.0,
        rd_progress=100.0,
    )

    # Easy Survival → alive (cash > 0, employees > 2, csat > 5) → 1.0
    assert grade_easy_survival(state) == 1.0

    # Medium Profit: 5k / 10k = 0.5
    assert grade_medium_profit(state) == 0.5

    # Hard Utopia: morale=1.0, rd=1.0, cash=1k → cash_score≈0.067
    # total ≈ 0.40×1 + 0.30×1 + 0.30×0.067 = 0.72
    score = grade_hard_utopia(state)
    assert 0.70 < score < 0.75, f"Unexpected hard utopia score: {score}"


def test_full_episode_does_not_crash():
    """Run a full episode without errors (smoke test)."""
    from agent.business_agent import CorporateAgent

    env = CEOEnvironment()
    agent = CorporateAgent()
    obs = env.reset(seed=7)

    for _ in range(50):
        action_np = agent.compute_action(obs.to_array())
        action = Action(
            price_adjustment=float(action_np[0]),
            marketing_push=float(np.clip(action_np[1], 0, 1)),
            hire_fire=float(action_np[2]),
            rd_investment=float(np.clip(action_np[3], 0, 1)),
            salary_adjustment=float(action_np[4]),
            task_allocation=float(action_np[5]),
            crisis_response=float(action_np[6]),
            budget_shift=float(action_np[7]),
        )
        obs = env.step(action)
        if obs.done:
            break

    assert env.typed_state().quarter > 0
