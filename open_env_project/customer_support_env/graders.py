"""
customer_support_env/graders.py — Deterministic graders with Business Metrics.

FIX APPLIED: All scores are now clamped to [0.01, 0.99] using _clamp_score()
to satisfy the OpenEnv requirement that grader outputs must be strictly in (0, 1).
"""

from __future__ import annotations
from typing import Any, Dict, List

_HELPFUL_KWS = frozenset(["refund", "help", "solution", "resolve", "assist", "fix", "support", "understand", "process", "issue", "address"])
_POLITE_KWS = frozenset(["sorry", "apologize", "apologies", "thank", "appreciate", "please"])

# FIX: Clamping function — maps [0, 1] → [0.01, 0.99]
SCORE_MIN = 0.01
SCORE_MAX = 0.99

def _clamp_score(raw: float) -> float:
    """Clamp a raw score to strict (0, 1) via linear transform."""
    clipped = max(0.0, min(1.0, raw))
    return SCORE_MIN + clipped * (SCORE_MAX - SCORE_MIN)

def _reply_scores(content: str) -> tuple[float, float]:
    c = content.lower()
    # FIX: Changed from binary 1.0/0.0 to smooth 0.95/0.05
    helpful = 0.95 if any(k in c for k in _HELPFUL_KWS) else 0.05
    polite = 0.95 if any(k in c for k in _POLITE_KWS) else 0.05
    return (helpful, polite)

def grade_easy(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """Easy: Categorize ticket correctly. FIX: Returns clamped scores."""
    for step in episode_history:
        if step.get("action_type") == "categorize":
            if step.get("content", "").strip().lower() == step.get("true_category", ""):
                return _clamp_score(1.0)  # FIX: was 1.0
            return _clamp_score(0.0)  # FIX: was 0.0
    return _clamp_score(0.0)  # FIX: was 0.0

def grade_medium(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """Medium: Helpful and polite response. FIX: Returns clamped scores."""
    reply_steps = [s for s in episode_history if s.get("action_type") == "reply"]
    if not reply_steps:
        return _clamp_score(0.0)  # FIX: was 0.0
    h, p = _reply_scores(reply_steps[-1].get("content", ""))
    raw_score = 0.6 * h + 0.4 * p
    return _clamp_score(raw_score)

def grade_hard(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """Hard: Full resolution, high satisfaction, minimized cost. FIX: Returns clamped scores."""
    if not episode_history:
        return _clamp_score(0.0)  # FIX: was 0.0

    # 1. Classification (30%) — FIX: smooth defaults
    class_score = 0.05
    for step in episode_history:
        if step.get("action_type") == "categorize":
            class_score = 0.95 if step.get("content", "").strip().lower() == step.get("true_category", "") else 0.05
            break

    # 2. Response Quality (40%) — FIX: smooth defaults
    reply_steps = [s for s in episode_history if s.get("action_type") == "reply"]
    resp_score = 0.05
    if reply_steps:
        h, p = _reply_scores(reply_steps[-1].get("content", ""))
        resp_score = min(1.0, 0.6 * h + 0.4 * p)

    # 3. Business Metrics (30%)
    last_step = episode_history[-1]
    biz_score = last_step.get("satisfaction_score", 0.5)  # FIX: default 0.5 instead of 0.0

    raw_score = 0.30 * class_score + 0.40 * resp_score + 0.30 * biz_score
    return _clamp_score(raw_score)

GRADERS: Dict[str, Any] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
