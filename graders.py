"""
graders.py — Episode-history aware deterministic graders for all OpenEnv tasks.
Strictly enforced (0, 1) continuous scoring for hackathon validation.
"""

from typing import List, Dict, Any, Optional

def smooth_score(score: float) -> float:
    """
    Enforces the "Strictly between 0 and 1" rule.
    Maps [0, 1] -> [0.01, 0.99] to satisfy the continuous value requirement.
    """
    return 0.01 + (min(1.0, max(0.0, score)) * 0.98)

# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: 🟢 Revenue Target Achievement
# ──────────────────────────────────────────────────────────────────────────────
def grade_revenue_target(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Objective: Increase revenue by 10% within 4 fiscal quarters.
    Logic: (Revenue Ratio achieved x 0.7) + (Survival Duration x 0.3)
    
    Success / Failure Criteria:
    - Poor (0.01 - 0.3): Revenue shrinks or stays flat; company bankrupts early.
    - Partial (0.3 - 0.7): Revenue grows but doesn't hit +10%, or survives but metrics flatline.
    - Strong (0.7 - 0.99): Achieves >= 10% revenue growth safely through all quarters.
    """
    if not episode_history: return smooth_score(0.0)
    
    initial_revenue = max(1.0, episode_history[0].get("Revenue", 5000))
    max_revenue = max([s.get("Revenue", 0) for s in episode_history])
    target_revenue = initial_revenue * 1.10
    
    # Progress towards +10% goal
    revenue_score = min(1.0, max_revenue / target_revenue)
    # Survival ratio (up to 4 quarters)
    survival_score = min(len(episode_history), 4) / 4.0
    
    raw_score = (revenue_score * 0.7) + (survival_score * 0.3)
    return smooth_score(raw_score)

# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: 🟡 Balanced Budget & Scaling
# ──────────────────────────────────────────────────────────────────────────────
def grade_budget_balance(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Objective: Maintain balanced budget (>10% per dept) while scaling team (>25).
    Logic: (Dept Funding Stability x 0.6) + (Workforce Growth x 0.4)
    
    Success / Failure Criteria:
    - Poor (0.01 - 0.3): Budgets severely skewed, workforce shrinks.
    - Partial (0.3 - 0.7): Reaches ~25 employees but budgets unstable, or stable but no growth.
    - Strong (0.7 - 0.99): Scales >25 staff with perfect >10% funding across all 5 departments.
    """
    if not episode_history: return smooth_score(0.0)
    
    avg_funding_ratio = sum([s.get("Departments_Funded", 0) / 5.0 for s in episode_history]) / len(episode_history)
    max_employees = max([s.get("Total Employees", 0) for s in episode_history])
    workforce_score = min(1.0, max_employees / 30.0) # Target 30 employees
    
    raw_score = (avg_funding_ratio * 0.6) + (workforce_score * 0.4)
    return smooth_score(raw_score)

# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: 🔴 Long-Term Strategic Growth
# ──────────────────────────────────────────────────────────────────────────────
def grade_strategic_growth(episode_history: List[Dict[str, Any]], seed: int = 42) -> float:
    """
    Objective: Maximize valuation and market position through R&D and Brand.
    Logic: (Valuation Growth x 0.5) + (Innovation/Reputation x 0.5)
    
    Success / Failure Criteria:
    - Poor (0.01 - 0.3): Valuation drops significantly, R&D ignored, reputation plummets.
    - Partial (0.3 - 0.7): Modest valuation growth; some R&D progress but low satisfaction.
    - Strong (0.7 - 0.99): Valuation hits $500k target, R&D progress near 100, high brand trust.
    """
    if not episode_history: return smooth_score(0.0)
    
    final_state = episode_history[-1]
    valuation_score = min(1.0, final_state.get("Valuation", 0) / 500000.0) # Target $500k
    
    innovation_score = final_state.get("RD_Progress", 0) / 100.0
    reputation_score = final_state.get("Customer Satisfaction", 0) / 100.0
    strategy_score = (innovation_score + reputation_score) / 2.0
    
    raw_score = (valuation_score * 0.5) + (strategy_score * 0.5)
    return smooth_score(raw_score)

# ──────────────────────────────────────────────────────────────────────────────
# OpenEnv Rubric & Registry
# ──────────────────────────────────────────────────────────────────────────────
from openenv.core.rubrics import Rubric

class CEORubric(Rubric):
    def __init__(self, task_id: str = "easy_revenue_target"):
        super().__init__()
        self.task_id = task_id
        self.grader = GRADERS.get(task_id, grade_revenue_target)

    def forward(self, action: Any, observation: Any) -> float:
        return 0.0

    def score_history(self, history: List[Dict[str, Any]]) -> float:
        return self.grader(history)

GRADERS: Dict[str, Any] = {
    "easy_revenue_target": grade_revenue_target,
    "medium_budget_balance": grade_budget_balance,
    "hard_strategic_growth": grade_strategic_growth,
}
