from models import State

def grade_easy_survival(state: State) -> float:
    """
    Easy Task: Survival.
    Score = 1.0 if cash remains > 0 and bankruptcy avoided after 100 quarters.
    Otherwise 0.0.
    """
    if state.cash > 0 and state.total_employees > 2 and state.customer_satisfaction > 5:
        return 1.0
    return 0.0

def grade_medium_profit(state: State) -> float:
    """
    Medium Task: Profit Target 10K.
    Score = min(1.0, max(0.0, final_profit / 10,000)).
    """
    return min(1.0, max(0.0, state.profit / 10000.0))

def grade_hard_utopia(state: State) -> float:
    """
    Hard Task: Utopia.
    High bar: Morale > 90%, R&D = 100%, Cash > 15,000. 
    Graded based on a weighted sum of these metrics.
    """
    morale_score = min(1.0, state.employee_morale / 90.0)
    rd_score = min(1.0, state.rd_progress / 100.0)
    cash_score = min(1.0, max(0.0, state.cash / 15000.0))
    
    # Needs a bit of all to succeed fully
    total_score = (morale_score * 0.4) + (rd_score * 0.3) + (cash_score * 0.3)
    return min(1.0, max(0.0, total_score))

TASKS = {
    "easy": grade_easy_survival,
    "medium": grade_medium_profit,
    "hard": grade_hard_utopia,
}
