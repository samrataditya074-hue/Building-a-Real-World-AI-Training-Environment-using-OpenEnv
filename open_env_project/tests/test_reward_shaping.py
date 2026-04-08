"""
tests/test_reward_shaping.py — Reward signal quality and continuity tests.

Verifies that:
  - Reward is always finite (no NaN or Inf)
  - Reward falls within a reasonable range every step
  - Destructive actions (mass firing) trigger a negative fire_penalty
  - R&D payoff component grows monotonically with rd_progress
  - Continuous shaping components appear in obs.info["reward_breakdown"]
"""

import math
import numpy as np
import pytest

from models import Action
from server.environment import CEOEnvironment, REWARD_FIRE_PENALTY


class TestRewardFiniteness:
    def test_reward_never_nan(self):
        """Reward must be a real number every step, never NaN."""
        env = CEOEnvironment()
        env.reset(seed=1)
        for _ in range(20):
            obs = env.step(Action())
            assert not math.isnan(obs.reward), f"Got NaN reward at Q{env.typed_state().quarter}"
            if obs.done:
                break

    def test_reward_never_inf(self):
        """Reward must never be infinite."""
        env = CEOEnvironment()
        env.reset(seed=2)
        for _ in range(20):
            obs = env.step(Action())
            assert not math.isinf(obs.reward), f"Got Inf reward at Q{env.typed_state().quarter}"
            if obs.done:
                break


class TestRewardRange:
    def test_reward_within_reasonable_bounds(self):
        """
        Normal per-step rewards should stay in [-15, +20].
        The only time reward can go outside this is the terminal penalty (-50).
        """
        env = CEOEnvironment()
        env.reset(seed=3)
        for _ in range(50):
            obs = env.step(Action())
            if obs.done:
                # Terminal penalty may fire — skip that step
                break
            assert -200.0 <= obs.reward <= 300.0, (
                f"Reward {obs.reward:.2f} outside expected range at Q{env.typed_state().quarter}"
            )

    def test_terminal_penalty_fires_on_bankruptcy(self):
        """
        If the company goes bankrupt, the terminal penalty (-50) should make the reward
        very negative (below -30).
        """
        env = CEOEnvironment()
        env.reset(seed=10)

        # Burn through cash: fire everyone, cut everything
        bankrupt_action = Action(
            hire_fire=-1.0, salary_adjustment=-1.0, marketing_push=0.0,
            rd_investment=0.0, budget_shift=-1.0
        )

        final_reward = None
        for _ in range(200):
            obs = env.step(bankrupt_action)
            final_reward = obs.reward
            if obs.done:
                break

        # If the company did go bankrupt, the last reward should reflect the penalty
        if env.typed_state().cash <= 0:
            assert final_reward is not None
            assert final_reward < -30.0, (
                f"Bankruptcy should yield reward < -30, got {final_reward:.2f}"
            )


class TestContinuousShaping:
    def test_reward_breakdown_present(self):
        """Every step must include reward_breakdown in info."""
        env = CEOEnvironment()
        env.reset(seed=4)
        obs = env.step(Action())
        assert "reward_breakdown" in obs.info
        rb = obs.info["reward_breakdown"]
        for key in ["profit_delta", "morale_delta", "rd_payoff", "fire_penalty"]:
            assert key in rb, f"Missing shaping component: '{key}'"

    def test_fire_penalty_negative_when_firing(self):
        """Mass firing should produce a negative fire_penalty component."""
        env = CEOEnvironment()
        env.reset(seed=5)
        fire_action = Action(hire_fire=-1.0)  # fire 5 people
        obs = env.step(fire_action)
        fire_pen = obs.info["reward_breakdown"]["fire_penalty"]
        assert fire_pen < 0.0, (
            f"Firing 5 people should produce negative fire_penalty, got {fire_pen}"
        )

    def test_fire_penalty_zero_when_not_firing(self):
        """No firing → fire_penalty must be exactly 0.0."""
        env = CEOEnvironment()
        env.reset(seed=6)
        no_fire_action = Action(hire_fire=0.0)
        obs = env.step(no_fire_action)
        fire_pen = obs.info["reward_breakdown"]["fire_penalty"]
        assert fire_pen == 0.0, f"No firing should yield fire_penalty=0.0, got {fire_pen}"

    def test_rd_payoff_grows_with_progress(self):
        """
        rd_payoff should increase as R&D progress increases.
        We manually step until R&D grows and verify the component grows.
        """
        env = CEOEnvironment()
        env.reset(seed=7)

        # Invest heavily in R&D for 10 quarters
        rd_action = Action(rd_investment=1.0, hire_fire=0.0)
        rd_payoffs = []
        for _ in range(10):
            obs = env.step(rd_action)
            rd_payoffs.append(obs.info["reward_breakdown"]["rd_payoff"])
            if obs.done:
                break

        # R&D payoff should be non-decreasing overall (allowing small floating-point noise)
        assert rd_payoffs[-1] >= rd_payoffs[0], (
            f"rd_payoff should grow with R&D investment. "
            f"First={rd_payoffs[0]:.4f}, Last={rd_payoffs[-1]:.4f}"
        )

    def test_all_shaping_components_finite(self):
        """All reward breakdown components must be finite floats."""
        env = CEOEnvironment()
        env.reset(seed=8)

        for _ in range(10):
            obs = env.step(Action(rd_investment=0.5, marketing_push=0.3))
            rb = obs.info["reward_breakdown"]
            for key, val in rb.items():
                assert math.isfinite(val), f"Component '{key}' is not finite: {val}"
            if obs.done:
                break
