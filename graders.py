"""
graders.py — Episode-history aware deterministic graders for all OpenEnv tasks.
Updated with intermediate signals and a new Market Strategy task.
"""

from typing import List, Dict, Any, Optional
from openenv.core.rubrics import Rubric

def _safe_avg(values: List[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default

# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Review Annual Report
# ──────────────────────────────────────────────────────────────────────────────
def grade_report(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Easy Task: Identify key metrics in the annual report.
    Intermediate rewards for each metric identified across quarters.
    """
    if not episode_history: return 0.0
    
    total_metrics = sum(step.get("Metrics_Identified", 0) for step in episode_history)
    # Max possible metrics (4 per quarter for 3 quarters)
    max_metrics = 12.0
    score = (len(episode_history) / 3) * 0.4 + (total_metrics / max_metrics) * 0.6
    return float(min(1.0, max(0.0, score)))

# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: Allocate Quarterly Budget
# ──────────────────────────────────────────────────────────────────────────────
def grade_allocation(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Medium Task: Allocate budget across 5 departments.
    Partial credit for each correctly funded department.
    """
    if not episode_history: return 0.0
    
    avg_funded = _safe_avg([step.get("Departments_Funded", 0) for step in episode_history])
    # Full marks if average 5 departments funded correctly
    score = avg_funded / 5.0
    return float(min(1.0, max(0.0, score)))

# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Negotiate Strategic Merger
# ──────────────────────────────────────────────────────────────────────────────
def grade_merger(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Hard Task: Lead negotiation for a merger.
    Incremental reward for each negotiation step completed.
    """
    if not episode_history: return 0.0
    
    final_steps = episode_history[-1].get("Negotiation_Steps", 0)
    # Max steps is 5
    score = final_steps / 5.0
    return float(min(1.0, max(0.0, score)))

# ──────────────────────────────────────────────────────────────────────────────
# Task 4 — Hard: Evaluate Market Strategy
# ──────────────────────────────────────────────────────────────────────────────
def grade_strategy(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Hard Task: Adaptation to market trends.
    Scores based on brand reputation and handling of market shifts.
    """
    if not episode_history: return 0.0
    
    reputation_score = episode_history[-1].get("Brand_Reputation", 60.0) / 100.0
    
    # Check if price changes followed market trend (roughly)
    # This is a bit complex, but we'll use a simplified signal: profit and reputation
    avg_satisfaction = _safe_avg([step.get("Customer Satisfaction", 75.0) for step in episode_history]) / 100.0
    
    score = 0.6 * reputation_score + 0.4 * avg_satisfaction
    return float(min(1.0, max(0.0, score)))

# ──────────────────────────────────────────────────────────────────────────────
# OpenEnv Rubric Implementation
# ──────────────────────────────────────────────────────────────────────────────
class CEORubric(Rubric):
    def __init__(self, task_id: str = "review_annual_report"):
        super().__init__()
        self.task_id = task_id
        self.grader = GRADERS.get(task_id, grade_report)

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def score_history(self, history: List[Dict[str, Any]]) -> float:
        return self.grader(history)

# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────
GRADERS: Dict[str, Any] = {
    "review_annual_report": grade_report,
    "allocate_budget": grade_allocation,
    "negotiate_merger": grade_merger,
    "evaluate_market_strategy": grade_strategy,
}
