"""
graders.py — Episode-history aware deterministic graders for all 3 OpenEnv tasks.

Unlike tasks.py (which grades the final State snapshot), these graders receive the
full episode history — a list of per-step metric dicts recorded by CEOEnvironment.
This allows them to measure things like "how many quarters was profit positive?"
rather than just looking at the last quarter.

All graders:
  - Accept (episode_history: list[dict], seed: int) for determinism documentation
  - Return a float in [0.0, 1.0]
  - Are pure functions — same inputs always produce the same output

Usage:
    from graders import GRADERS
    score = GRADERS["easy"](env.state().metrics_history, seed=42)
"""

from __future__ import annotations

from typing import List, Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────
def _safe_avg(values: List[float], default: float = 0.0) -> float:
    """Returns the mean of a list, or default if the list is empty."""
    return sum(values) / len(values) if values else default


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Survive 4 Quarters
# ──────────────────────────────────────────────────────────────────────────────
def grade_easy(
    episode_history: List[Dict[str, Any]],
    seed: int = 42,  # kept for API consistency; grader is deterministic regardless
) -> float:
    """
    Easy Task: Survive at least 4 quarters without going bankrupt.

    Score = min(quarters_survived, 4) / 4
      → 1.0 if the company made it through 4 quarters without termination.
      → 0.75 if it survived 3 quarters, etc.

    Plain English:
      Just keep the lights on for one year (4 business quarters).
      Even a small profit each quarter is enough to score full marks.

    Args:
        episode_history: List of per-step metric dicts from State.metrics_history.
                         Each dict has keys: Quarter, Cash, Revenue, Profit, etc.
        seed: RNG seed used during the episode (for reproducibility tracking).

    Returns:
        Float in [0.0, 1.0].
    """
    TARGET_QUARTERS = 4
    quarters_survived = len(episode_history)  # each entry = one quarter completed
    score = min(quarters_survived, TARGET_QUARTERS) / TARGET_QUARTERS
    return float(min(1.0, max(0.0, score)))


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

    Score = 0.6 × valuation_growth_component + 0.4 × morale_component

    Where:
      valuation_growth_component = min(1.0, growth_pct / 20)
        → 1.0 if valuation grew by 20%+, partial credit below that.
      morale_component = avg_morale / 100

    Plain English:
      Grow the company's overall worth (cash + brand + people + tech),
      AND keep employees happy while doing it.

    Args:
        episode_history: List of per-step dicts from State.metrics_history.
        seed: RNG seed used during the episode.

    Returns:
        Float in [0.0, 1.0].
    """
    if not episode_history:
        return 0.0

    TARGET_GROWTH_PCT = 20.0  # 20% valuation growth = full marks on that component
    MORALE_WEIGHT = 0.4
    VALUATION_WEIGHT = 0.6

    initial_valuation = episode_history[0].get("Valuation", 1.0)
    final_valuation = episode_history[-1].get("Valuation", initial_valuation)

    # Avoid division by zero if initial valuation is 0
    if initial_valuation <= 0:
        growth_pct = 0.0
    else:
        growth_pct = ((final_valuation - initial_valuation) / initial_valuation) * 100.0

    valuation_component = min(1.0, max(0.0, growth_pct / TARGET_GROWTH_PCT))

    # Average morale across all quarters (stored as 0–100, divide to get 0–1)
    morale_values = [
        step.get("Morale", 50.0) / 100.0
        for step in episode_history
    ]
    morale_component = _safe_avg(morale_values, default=0.0)

    score = (VALUATION_WEIGHT * valuation_component) + (MORALE_WEIGHT * morale_component)
    return float(min(1.0, max(0.0, score)))


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Win the Pricing War for 8 Profitable Quarters
# ──────────────────────────────────────────────────────────────────────────────
def grade_hard(
    episode_history: List[Dict[str, Any]],
    seed: int = 42,
) -> float:
    """
    Hard Task: In at least 6 of the first 8 quarters, simultaneously:
      (A) underprice the competitor (your price < competitor price), AND
      (B) maintain positive profit (profit > 0)
    … while also keeping average morale healthy.

    Score = 0.5 × (pricing_wins / 8)
           + 0.3 × (profit_wins / 8)
           + 0.2 × (avg_morale / 100)

    Plain English:
      Beat the competitor on price to capture market share, but don't
      discount so hard that you lose money. Do this consistently for
      8 quarters straight while keeping your team motivated.

    Args:
        episode_history: List of per-step dicts. Expects keys:
            "Our_Price" (float), "Competitor_Price" (float),
            "Profit" (float), "Morale" (float).
        seed: RNG seed used during the episode.

    Returns:
        Float in [0.0, 1.0].
    """
    TARGET_QUARTERS = 8
    PRICING_WEIGHT = 0.5
    PROFIT_WEIGHT = 0.3
    MORALE_WEIGHT = 0.2

    if not episode_history:
        return 0.0

    # Only look at first 8 quarters (or however many exist)
    evaluation_window = episode_history[:TARGET_QUARTERS]

    pricing_wins = 0
    profit_wins = 0
    morale_values: List[float] = []

    for step in evaluation_window:
        our_price = step.get("Our_Price", 50.0)
        comp_price = step.get("Competitor_Price", 50.0)
        profit = step.get("Profit", 0.0)
        morale = step.get("Morale", 50.0)

        # Pricing win: we are cheaper than the competitor
        if our_price < comp_price:
            pricing_wins += 1

        # Profit win: we made money this quarter
        if profit > 0:
            profit_wins += 1

        morale_values.append(morale / 100.0)

    pricing_component = pricing_wins / TARGET_QUARTERS
    profit_component = profit_wins / TARGET_QUARTERS
    morale_component = _safe_avg(morale_values, default=0.0)

    score = (
        PRICING_WEIGHT * pricing_component
        + PROFIT_WEIGHT * profit_component
        + MORALE_WEIGHT * morale_component
    )
    return float(min(1.0, max(0.0, score)))


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
GRADERS: Dict[str, Any] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
