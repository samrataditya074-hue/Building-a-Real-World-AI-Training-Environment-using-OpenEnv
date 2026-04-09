"""
graders.py — Episode-history aware deterministic graders for all 3 OpenEnv tasks.

Unlike tasks.py (which grades the final State snapshot), these graders receive the
full episode history — a list of per-step metric dicts recorded by CEOEnvironment.
This allows them to measure things like "how many quarters was profit positive?"
rather than just looking at the last quarter.

All graders:
  - Accept (episode_history: list[dict], seed: int) for determinism documentation
  - Return a float in (0.0, 1.0) — STRICTLY between 0 and 1, never exactly 0 or 1
  - Are pure functions — same inputs always produce the same output

FIX APPLIED: All scores are now clamped to [0.01, 0.99] using _clamp_score()
to satisfy the OpenEnv requirement that grader outputs must be strictly in (0, 1).

Usage:
    from graders import GRADERS
    score = GRADERS["easy"](env.state().metrics_history, seed=42)
"""

from __future__ import annotations

from typing import List, Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_avg(values: List[float], default: float = 0.0) -> float:
    """Returns the mean of a list, or default if the list is empty."""
    return sum(values) / len(values) if values else default


# FIX: Clamping function to ensure scores are strictly in (0, 1).
# Maps any raw score from [0.0, 1.0] into [0.01, 0.99] using a linear transform.
# This guarantees the output is NEVER exactly 0 or exactly 1.
SCORE_MIN = 0.01
SCORE_MAX = 0.99

def _clamp_score(raw: float) -> float:
    """
    Clamp a raw score to the strict open interval (0, 1).

    Maps [0.0, 1.0] → [0.01, 0.99] via: result = 0.01 + (clipped_raw * 0.98)
    This ensures:
      - A perfect raw score of 1.0 returns 0.99
      - A zero raw score of 0.0 returns 0.01
      - All intermediate values scale proportionally
    """
    clipped = max(0.0, min(1.0, raw))
    return SCORE_MIN + clipped * (SCORE_MAX - SCORE_MIN)


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Survive 4 Quarters
# ──────────────────────────────────────────────────────────────────────────────
def grade_easy(
    episode_history: List[Dict[str, Any]],
    seed: int = 42,
) -> float:
    """
    Easy Task: Survive at least 4 quarters without going bankrupt.
    Raw Score = min(quarters_survived, 4) / 4
    FIX: Final score is clamped via _clamp_score() so it never returns
    exactly 0.0 or 1.0.

    Returns:
        Float in (0.0, 1.0) — strictly between 0 and 1.
    """
    TARGET_QUARTERS = 4
    quarters_survived = len(episode_history)
    raw_score = min(quarters_survived, TARGET_QUARTERS) / TARGET_QUARTERS
    return _clamp_score(raw_score)


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: Grow Valuation While Keeping Morale High
# ──────────────────────────────────────────────────────────────────────────────
def grade_medium(
    episode_history: List[Dict[str, Any]],
    seed: int = 42,
) -> float:
    """
    Medium Task: Grow company valuation by ≥20% over 50 quarters while
    keeping average employee morale above 60%.
    Raw Score = 0.6 × valuation_growth_component + 0.4 × morale_component
    FIX: Final score is clamped via _clamp_score().

    Returns:
        Float in (0.0, 1.0) — strictly between 0 and 1.
    """
    # FIX: Empty history returns bounded minimum instead of exact 0.0
    if not episode_history:
        return _clamp_score(0.0)

    TARGET_GROWTH_PCT = 20.0
    MORALE_WEIGHT = 0.4
    VALUATION_WEIGHT = 0.6

    initial_valuation = episode_history[0].get("Valuation", 1.0)
    final_valuation = episode_history[-1].get("Valuation", initial_valuation)

    if initial_valuation <= 0:
        growth_pct = 0.0
    else:
        growth_pct = ((final_valuation - initial_valuation) / initial_valuation) * 100.0

    valuation_component = min(1.0, max(0.0, growth_pct / TARGET_GROWTH_PCT))

    morale_values = [
        step.get("Morale", 50.0) / 100.0
        for step in episode_history
    ]
    morale_component = _safe_avg(morale_values, default=0.0)

    raw_score = (VALUATION_WEIGHT * valuation_component) + (MORALE_WEIGHT * morale_component)
    return _clamp_score(raw_score)


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Win the Pricing War for 8 Profitable Quarters
# ──────────────────────────────────────────────────────────────────────────────
def grade_hard(
    episode_history: List[Dict[str, Any]],
    seed: int = 42,
) -> float:
    """
    Hard Task: Underprice competitor while staying profitable for 8 quarters.
    Raw Score = 0.5 × (pricing_wins / 8) + 0.3 × (profit_wins / 8) + 0.2 × (avg_morale / 100)
    FIX: Final score is clamped via _clamp_score().

    Returns:
        Float in (0.0, 1.0) — strictly between 0 and 1.
    """
    TARGET_QUARTERS = 8
    PRICING_WEIGHT = 0.5
    PROFIT_WEIGHT = 0.3
    MORALE_WEIGHT = 0.2

    # FIX: Empty history returns bounded minimum instead of exact 0.0
    if not episode_history:
        return _clamp_score(0.0)

    evaluation_window = episode_history[:TARGET_QUARTERS]

    pricing_wins = 0
    profit_wins = 0
    morale_values: List[float] = []

    for step in evaluation_window:
        our_price = step.get("Our_Price", 50.0)
        comp_price = step.get("Competitor_Price", 50.0)
        profit = step.get("Profit", 0.0)
        morale = step.get("Morale", 50.0)

        if our_price < comp_price:
            pricing_wins += 1

        if profit > 0:
            profit_wins += 1

        morale_values.append(morale / 100.0)

    pricing_component = pricing_wins / TARGET_QUARTERS
    profit_component = profit_wins / TARGET_QUARTERS
    morale_component = _safe_avg(morale_values, default=0.0)

    raw_score = (
        PRICING_WEIGHT * pricing_component
        + PROFIT_WEIGHT * profit_component
        + MORALE_WEIGHT * morale_component
    )
    return _clamp_score(raw_score)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# FIX: Keys are "easy", "medium", "hard" to match evaluator expectations.
# ──────────────────────────────────────────────────────────────────────────────
GRADERS: Dict[str, Any] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    # Backward-compatible aliases
    "survive_easy": grade_easy,
    "grow_val_medium": grade_medium,
    "undercut_hard": grade_hard,
}
