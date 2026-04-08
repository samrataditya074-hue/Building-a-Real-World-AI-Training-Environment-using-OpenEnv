"""
customer_support_env/graders.py — Deterministic graders with Business Metrics.
"""

from __future__ import annotations
from typing import Any, Dict, List

_HELPFUL_KWS = frozenset(["refund", "help", "solution", "resolve", "assist", "fix", "support", "understand", "process", "issue", "address"])
_POLITE_KWS = frozenset(["sorry", "apologize", "apologies", "thank", "appreciate", "please"])

def _reply_scores(content: str) -> tuple[float, float]:
    c = content.lower()
    return (1.0 if any(k in c for k in _HELPFUL_KWS) else 0.0, 1.0 if any(k in c for k in _POLITE_KWS) else 0.0)

def grade_easy(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """Easy: Categorize ticket correctly."""
    for step in episode_history:
        if step.get("action_type") == "categorize":
            if step.get("content", "").strip().lower() == step.get("true_category", ""):
                return 1.0
            return 0.0
    return 0.0

def grade_medium(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """Medium: Helpful and polite response."""
    reply_steps = [s for s in episode_history if s.get("action_type") == "reply"]
    if not reply_steps:
        return 0.0
    h, p = _reply_scores(reply_steps[-1].get("content", ""))
    return float(min(1.0, max(0.0, 0.6 * h + 0.4 * p)))

def grade_hard(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """Hard: Full resolution, high satisfaction, minimized cost."""
    if not episode_history:
        return 0.0

    # 1. Classification (30%)
    class_score = 0.0
    for step in episode_history:
        if step.get("action_type") == "categorize":
            class_score = 1.0 if step.get("content", "").strip().lower() == step.get("true_category", "") else 0.0
            break

    # 2. Response Quality (40%)
    reply_steps = [s for s in episode_history if s.get("action_type") == "reply"]
    resp_score = 0.0
    if reply_steps:
        h, p = _reply_scores(reply_steps[-1].get("content", ""))
        resp_score = min(1.0, 0.6 * h + 0.4 * p)

    # 3. Business Metrics (30%)
    # Use satisfaction_score directly (which naturally drops from unnecessary esc/repeats)
    last_step = episode_history[-1]
    biz_score = last_step.get("satisfaction_score", 0.0)

    score = 0.30 * class_score + 0.40 * resp_score + 0.30 * biz_score
    return float(min(1.0, max(0.0, score)))

GRADERS: Dict[str, Any] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
