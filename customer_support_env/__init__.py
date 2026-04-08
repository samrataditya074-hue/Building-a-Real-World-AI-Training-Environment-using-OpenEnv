"""
customer_support_env — OpenEnv-compliant Customer Support Ticket Resolution Environment.

Exports:
    SupportEnvironment  — the main environment class
    Action              — Pydantic action model
    Observation         — Pydantic observation model
    State               — dataclass for full internal state
"""
from customer_support_env.models import Action, Observation, State
from customer_support_env.environment import SupportEnvironment

__all__ = ["SupportEnvironment", "Action", "Observation", "State"]
