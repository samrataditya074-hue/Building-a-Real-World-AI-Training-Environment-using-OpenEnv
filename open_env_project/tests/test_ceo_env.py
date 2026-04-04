import pytest
import numpy as np

from models import Action, Observation, State
from server.environment import CEOEnvironment
from tasks import grade_easy_survival, grade_medium_profit, grade_hard_utopia

def test_environment_reset():
    env = CEOEnvironment()
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert 0.0 <= obs.cash_norm <= 2.0  # approximate boundary
    
    state = env.state()
    assert state.cash == 10000.0
    assert state.total_employees == 20

def test_environment_step():
    env = CEOEnvironment()
    env.reset()
    action = Action(price_adjustment=0.1, hire_fire=0.5)  # Hire some people
    
    result = env.step(action)
    assert not result.done
    assert "pos_reward" in result.info
    assert "thought" in result.info
    
    state = env.state()
    assert state.total_employees > 20  # Should have hired employees
    assert state.quarter == 1

def test_graders():
    state = State(cash=1000, total_employees=5, customer_satisfaction=50, profit=5000, employee_morale=95, rd_progress=100)
    
    # Easy Survival -> True
    assert grade_easy_survival(state) == 1.0
    
    # Medium Profit 5k/10k -> 0.5
    assert grade_medium_profit(state) == 0.5
    
    # Hard Utopia (morale 1.0, rd 1.0, cash 1k=~0.06 -> 0.4*1 + 0.3*1 + 0.3*0.06 ~ 0.72)
    score = grade_hard_utopia(state)
    assert 0.7 < score < 0.75
