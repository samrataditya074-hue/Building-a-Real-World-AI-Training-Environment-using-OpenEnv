"""
customer_support_env/models.py — Data blueprints for the Hybrid Support Environment.

Plain-English overview:
  - Action      : what the agent does each step (categorize / reply / escalate)
  - Observation : what the agent sees — ticket state + business metrics
  - Reward      : structured breakdown of per-step score
  - State       : full internal environment state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Action (Pydantic model)
# ──────────────────────────────────────────────────────────────────────────────
class Action(BaseModel):
    """
    The agent's decision for one step.

    action_type:
        "categorize" — label the ticket with a category string
        "reply"      — send a text response to the customer
        "escalate"   — escalate the ticket (incurs business cost)

    content:
        text payload for the chosen action
    """

    action_type: Literal["categorize", "reply", "escalate"] = Field(
        description='Action type: "categorize", "reply", or "escalate".'
    )
    content: str = Field(
        min_length=1,
        max_length=1000,
        description="Text payload for the action.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ──────────────────────────────────────────────────────────────────────────────
# Observation (Pydantic model)
# ──────────────────────────────────────────────────────────────────────────────
class Observation(BaseModel):
    """
    The agent's view of the support ticket AND business metrics.
    """

    # ── Ticket state ──────────────────────────────────────────────────────────
    ticket_id: str = Field(description="Unique ticket ID.")
    customer_message: str = Field(description="Original customer message.")
    category: Optional[str] = Field(default=None, description="Assigned category.")
    conversation_history: List[str] = Field(default_factory=list, description="Message log.")
    urgency: str = Field(description="Urgency: 'low', 'medium', 'high'.")
    resolved: bool = Field(default=False, description="True when ticket is handled.")
    step_count: int = Field(default=0, description="Steps taken in this episode.")
    last_action_type: Optional[str] = Field(default=None, description="Previous action type.")

    # ── Business metrics ──────────────────────────────────────────────────────
    satisfaction_score: float = Field(default=0.5, description="Customer satisfaction (0.0 to 1.0).")
    cost_spent: float = Field(default=0.0, description="Cost incurred by operations (e.g. escalations).")
    tickets_resolved: int = Field(default=0, description="Total tickets resolved.")

    # ── OpenEnv standard fields ───────────────────────────────────────────────
    reward: float = Field(default=0.0, description="Reward received this step.")
    done: bool = Field(default=False, description="True if episode has ended.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Debugging info.")

    model_config = {"arbitrary_types_allowed": True}

    def to_array(self) -> np.ndarray:
        """
        Numeric encoding for RL consumption. 
        Shape: (11,) - ticket features + business metrics.
        """
        CATEGORY_MAP: Dict[Optional[str], float] = {
            None: 0.0, "login_issue": 1.0, "payment_failure": 2.0,
            "refund_request": 3.0, "delivery_delay": 4.0, "app_crash": 5.0,
        }
        URGENCY_MAP = {"low": 0.0, "medium": 0.5, "high": 1.0}
        ACTION_MAP: Dict[Optional[str], float] = {
            None: 0.0, "categorize": 1.0, "reply": 2.0, "escalate": 3.0,
        }

        return np.array([
            float(self.step_count) / 5.0,
            URGENCY_MAP.get(self.urgency, 0.5),
            1.0 if self.resolved else 0.0,
            CATEGORY_MAP.get(self.category, 0.0) / 5.0,
            min(1.0, float(len(self.conversation_history)) / 10.0),
            ACTION_MAP.get(self.last_action_type, 0.0) / 3.0,
            1.0 if self.category is not None else 0.0,
            min(1.0, float(len(self.customer_message)) / 500.0),
            self.satisfaction_score,
            min(1.0, self.cost_spent / 100.0),
            float(self.tickets_resolved) / 10.0
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class State:
    """Internal representation of the ticket workflow and business metrics."""

    ticket_id: str = ""
    customer_message: str = ""
    true_category: str = ""
    category: Optional[str] = None

    conversation_history: List[str] = field(default_factory=list)
    urgency: str = "medium"

    resolved: bool = False
    step_count: int = 0
    has_categorized: bool = False
    has_replied: bool = False
    has_escalated: bool = False
    last_action_type: Optional[str] = None
    
    # ── Business metrics ──────────────────────────────────────────────────────
    satisfaction_score: float = 0.5  # Starts neutral
    cost_spent: float = 0.0
    tickets_resolved: int = 0
    
    action_counts: Dict[str, int] = field(default_factory=dict)
    episode_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_observation(self) -> Observation:
        return Observation(
            ticket_id=self.ticket_id,
            customer_message=self.customer_message,
            category=self.category,
            conversation_history=list(self.conversation_history),
            urgency=self.urgency,
            resolved=self.resolved,
            step_count=self.step_count,
            last_action_type=self.last_action_type,
            satisfaction_score=self.satisfaction_score,
            cost_spent=self.cost_spent,
            tickets_resolved=self.tickets_resolved
        )
