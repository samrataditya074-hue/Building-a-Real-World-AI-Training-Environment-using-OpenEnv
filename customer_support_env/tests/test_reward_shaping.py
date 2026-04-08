"""
customer_support_env/tests/test_reward_shaping.py
"""
import pytest
from customer_support_env.models import Action
from customer_support_env.environment import SupportEnvironment

def test_reward_bounds():
    env = SupportEnvironment()
    env.reset()
    obs = env.step(Action(action_type="reply", content="This is absolutely horrible."))
    assert 0.0 <= obs.reward <= 1.0

def test_reward_correct_categorization():
    env = SupportEnvironment()
    obs = env.reset(seed=42) # known scenario
    act = Action(action_type="categorize", content=env.typed_state().true_category)
    obs = env.step(act)
    assert obs.info["reward_breakdown"].get("correct_categorize") == 0.3
    
def test_reward_polite_helpful():
    env = SupportEnvironment()
    env.reset(seed=42)
    act = Action(action_type="reply", content="I sincerely apologize for the issue. Let me help you resolve this.")
    obs = env.step(act)
    bd = obs.info["reward_breakdown"]
    assert bd.get("helpful_response") == 0.4
    assert bd.get("high_satisfaction") == 0.2  # initial 0.2-0.8 + bonuses push to >0.8
