"""
customer_support_env/tests/test_openenv_spec.py
"""
import pytest
import numpy as np

from customer_support_env.models import Action, Observation
from customer_support_env.environment import SupportEnvironment

def test_action_validation():
    act = Action(action_type="categorize", content="login_issue")
    assert act.action_type == "categorize"
    assert act.content == "login_issue"

def test_observation_array():
    obs = Observation(
        ticket_id="TKT", customer_message="Help", urgency="high",
        satisfaction_score=0.9, cost_spent=50.0, tickets_resolved=1
    )
    arr = obs.to_array()
    assert arr.shape == (11,)
    assert arr.dtype == np.float32

def test_environment_spec():
    env = SupportEnvironment()
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert not obs.done
    assert obs.step_count == 0
    assert "ticket_id" in env.state()
