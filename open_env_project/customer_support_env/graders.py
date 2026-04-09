"""
customer_support_env/graders.py — Deterministic graders with Business Metrics.

FIX APPLIED: All scores are now clamped to [0.01, 0.99] using _clamp_score()
to satisfy the OpenEnv requirement that grader outputs must be strictly in (0, 1).
Binary scoring (0 or 1) has been replaced with smooth bounded scoring.
"""

from __future__ import annotations
from typing import Any, Dict, List

_HELPFUL_KWS = frozenset(["refund", "help", "solution", "resolve", "assist", "fix", "support", "understand", "process", "issue", "address"])
_POLITE_KWS = frozenset(["sorry", "apologize", "apologies", "thank", "appreciate", "please"])


# ──────────────────────────────────────────────────────────────────────────────
# FIX: Score clamping function — ensures scores are strictly in (0, 1)
# Maps [0.0, 1.0] → [0.01, 0.99] via linear transform
# ──────────────────────────────────────────────────────────────────────────────
SCORE_MIN = 0.01
SCORE_MAX = 0.99

def _clamp_score(raw: float) -> float:
    """
    Clamp a raw score to the strict open interval (0, 1).
    Maps [0.0, 1.0] → [0.01, 0.99] via: result = 0.01 + (clipped_raw * 0.98)
    """
    clipped = max(0.0, min(1.0, raw))
    return SCORE_MIN + clipped * (SCORE_MAX - SCORE_MIN)


def _reply_scores(content: str) -> tuple[float, float]:
    c = content.lower()
    # FIX: Changed from binary 1.0/0.0 to smooth 0.95/0.05 scores
    # so that downstream computation never produces exact 0.0 or 1.0
    helpful = 0.95 if any(k in c for k in _HELPFUL_KWS) else 0.05
    polite = 0.95 if any(k in c for k in _POLITE_KWS) else 0.05
    return (helpful, polite)


def grade_easy(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Easy: Categorize ticket correctly.

    FIX: Instead of returning exact 0.0 or 1.0, returns clamped smooth scores.
    Correct categorization → 0.99, wrong → 0.01, no attempt → 0.01.
    """
    for step in episode_history:
        if step.get("action_type") == "categorize":
            if step.get("content", "").strip().lower() == step.get("true_category", ""):
                # FIX: Was return 1.0, now returns clamped maximum
                return _clamp_score(1.0)
            # FIX: Was return 0.0, now returns clamped minimum
            return _clamp_score(0.0)
    # FIX: Was return 0.0, now returns clamped minimum
    return _clamp_score(0.0)


def grade_medium(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Medium: Helpful and polite response.

    FIX: Uses smooth _reply_scores and clamps final output.
    """
    reply_steps = [s for s in episode_history if s.get("action_type") == "reply"]
    if not reply_steps:
        # FIX: Was return 0.0, now returns clamped minimum
        return _clamp_score(0.0)
    h, p = _reply_scores(reply_steps[-1].get("content", ""))
    raw_score = 0.6 * h + 0.4 * p
    # FIX: Clamp final score to strict (0, 1) range
    return _clamp_score(raw_score)


def grade_hard(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Hard: Full resolution, high satisfaction, minimized cost.

    FIX: All sub-scores use smooth values and final output is clamped.
    """
    if not episode_history:
        # FIX: Was return 0.0, now returns clamped minimum
        return _clamp_score(0.0)

    # 1. Classification (30%) — FIX: smooth scoring instead of binary 0/1
    class_score = 0.05  # FIX: default is no longer exact 0.0
    for step in episode_history:
        if step.get("action_type") == "categorize":
            # FIX: Was 1.0/0.0, now 0.95/0.05 for smooth scoring
            class_score = 0.95 if step.get("content", "").strip().lower() == step.get("true_category", "") else 0.05
            break

    # 2. Response Quality (40%) — FIX: uses smooth _reply_scores
    reply_steps = [s for s in episode_history if s.get("action_type") == "reply"]
    resp_score = 0.05  # FIX: default is no longer exact 0.0
    if reply_steps:
        h, p = _reply_scores(reply_steps[-1].get("content", ""))
        resp_score = min(1.0, 0.6 * h + 0.4 * p)

    # 3. Business Metrics (30%)
    # Use satisfaction_score directly (which naturally drops from unnecessary esc/repeats)
    last_step = episode_history[-1]
    biz_score = last_step.get("satisfaction_score", 0.5)  # FIX: default 0.5 instead of 0.0

    raw_score = 0.30 * class_score + 0.40 * resp_score + 0.30 * biz_score
    # FIX: Clamp final score to strict (0, 1) range
    return _clamp_score(raw_score)


GRADERS: Dict[str, Any] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
