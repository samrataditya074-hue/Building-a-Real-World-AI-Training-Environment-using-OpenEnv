"""
tests/test_graders.py — Deterministic grader correctness tests.

Verifies that all graders in graders.py:
  - Return floats in [0.0, 1.0]
  - Are fully deterministic (same inputs → same output)
  - Handle edge cases correctly (empty history, bankrupt run)
  - Return expected values for known episode fixtures
"""

import pytest
from typing import List, Dict, Any

from graders import grade_easy, grade_medium, grade_hard, GRADERS


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_step(q: int, cash: float = 200_000.0, profit: float = 1_000.0,
               valuation: float = 250_000.0, morale: float = 80.0,
               our_price: float = 40.0, comp_price: float = 50.0) -> Dict[str, Any]:
    """Helper: create a single episode step dict mimicking metrics_history format."""
    return {
        "Quarter": q,
        "Cash": cash,
        "Revenue": profit + 10_000,
        "Profit": profit,
        "Valuation": valuation,
        "Our_Price": our_price,
        "Competitor_Price": comp_price,
        "Total Employees": 20,
        "Morale": morale,
        "Customer Satisfaction": 75.0,
        "RD_Progress": 10.0,
        "Reward": 5.0,
        "Headline": "Test Quarter",
        "AI Thought": "Test thought",
    }


HEALTHY_4Q = [_make_step(q) for q in range(1, 5)]     # 4 healthy quarters
HEALTHY_50Q = [_make_step(q) for q in range(1, 51)]   # 50 healthy quarters
EMPTY: List[Dict[str, Any]] = []                        # no steps (bankrupt Q0)


# ─── Easy task ────────────────────────────────────────────────────────────────

class TestGradeEasy:
    def test_full_survival_scores_1(self):
        """Surviving all 4 quarters should return 1.0."""
        score = grade_easy(HEALTHY_4Q, seed=42)
        assert score == 1.0

    def test_partial_survival(self):
        """Surviving 2 of 4 quarters should return 0.5."""
        score = grade_easy(HEALTHY_4Q[:2], seed=42)
        assert abs(score - 0.5) < 1e-6

    def test_empty_history_returns_zero(self):
        """Empty history (died before completing any quarter) should return 0.0."""
        score = grade_easy(EMPTY, seed=42)
        assert score == 0.0

    def test_score_capped_at_one(self):
        """Surviving more than 4 quarters should not exceed 1.0."""
        score = grade_easy(HEALTHY_50Q, seed=42)
        assert score == 1.0

    def test_deterministic(self):
        """Calling twice with same args must return same result."""
        s1 = grade_easy(HEALTHY_4Q, seed=42)
        s2 = grade_easy(HEALTHY_4Q, seed=42)
        assert s1 == s2

    def test_in_range(self):
        for history in [EMPTY, HEALTHY_4Q, HEALTHY_50Q]:
            score = grade_easy(history, seed=42)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ─── Medium task ──────────────────────────────────────────────────────────────

class TestGradeMedium:
    def test_strong_growth_scores_high(self):
        """If valuation doubled (100% growth >> 20% target), expect close to 1.0."""
        history = [_make_step(q, valuation=200_000 + q * 10_000, morale=80)
                   for q in range(1, 51)]
        score = grade_medium(history, seed=42)
        assert score > 0.8, f"Expected high score, got {score}"

    def test_no_growth_gets_morale_credit(self):
        """If valuation is flat but morale is high, score should still be > 0."""
        flat_history = [_make_step(q, valuation=250_000, morale=90)
                        for q in range(1, 51)]
        score = grade_medium(flat_history, seed=42)
        assert score > 0.3, "High morale alone should yield partial score"

    def test_empty_history_returns_zero(self):
        score = grade_medium(EMPTY, seed=42)
        assert score == 0.0

    def test_deterministic(self):
        s1 = grade_medium(HEALTHY_50Q, seed=42)
        s2 = grade_medium(HEALTHY_50Q, seed=42)
        assert s1 == s2

    def test_in_range(self):
        for history in [EMPTY, HEALTHY_4Q, HEALTHY_50Q]:
            score = grade_medium(history, seed=42)
            assert 0.0 <= score <= 1.0


# ─── Hard task ────────────────────────────────────────────────────────────────

class TestGradeHard:
    def test_always_winning_pricing_and_profit(self):
        """
        If we undercut the competitor AND profit > 0 every quarter for 8 quarters,
        morale is also 80% → score should be high.
        """
        winning_history = [
            _make_step(q, profit=1_000, morale=80, our_price=40, comp_price=50)
            for q in range(1, 51)
        ]
        score = grade_hard(winning_history, seed=42)
        # pricing_wins=8/8=1, profit_wins=8/8=1, morale=0.8
        # score = 0.5*1 + 0.3*1 + 0.2*0.8 = 0.96
        assert score > 0.9, f"Expected high score, got {score}"

    def test_losing_pricing_and_profit(self):
        """If we always overprice AND lose money, score should be low."""
        losing_history = [
            _make_step(q, profit=-500, morale=50, our_price=60, comp_price=50)
            for q in range(1, 51)
        ]
        score = grade_hard(losing_history, seed=42)
        # pricing_wins=0, profit_wins=0, morale=0.5 → 0.2*0.5 = 0.1
        assert score < 0.15, f"Expected low score, got {score}"

    def test_empty_history_returns_zero(self):
        score = grade_hard(EMPTY, seed=42)
        assert score == 0.0

    def test_deterministic(self):
        s1 = grade_hard(HEALTHY_50Q, seed=42)
        s2 = grade_hard(HEALTHY_50Q, seed=42)
        assert s1 == s2

    def test_in_range(self):
        for history in [EMPTY, HEALTHY_4Q, HEALTHY_50Q]:
            score = grade_hard(history, seed=42)
            assert 0.0 <= score <= 1.0

    def test_only_first_8_quarters_evaluated(self):
        """Hard grader only looks at first 8 quarters — extra quarters should not change score."""
        base_8 = [_make_step(q, our_price=40, comp_price=50, profit=500)
                  for q in range(1, 9)]
        extended = base_8 + [_make_step(q, our_price=70, comp_price=50, profit=-1000)
                              for q in range(9, 51)]
        score_base = grade_hard(base_8, seed=42)
        score_extended = grade_hard(extended, seed=42)
        assert abs(score_base - score_extended) < 1e-6, (
            "Hard grader must only evaluate first 8 quarters"
        )


# ─── GRADERS registry ─────────────────────────────────────────────────────────

def test_graders_registry_complete():
    """GRADERS dict must contain all three task keys."""
    assert set(GRADERS.keys()) == {"easy", "medium", "hard"}

def test_all_graders_callable():
    for name, fn in GRADERS.items():
        assert callable(fn), f"GRADERS['{name}'] is not callable"
