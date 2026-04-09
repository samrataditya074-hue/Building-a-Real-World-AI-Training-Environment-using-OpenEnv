"""
customer_support_env/environment.py — Main simulation engine with Business Metrics.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from customer_support_env.models import Action, Observation, State

logger = logging.getLogger("support_env")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [SUPPORT-ENV] %(levelname)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("ENV_LOG_LEVEL", "ERROR").upper(), logging.ERROR))


# ── Reward constants ───────────────────────────────────────────────────────────
REWARD_CORRECT_CATEGORIZE: float = 0.30
REWARD_HELPFUL_REPLY: float = 0.40
REWARD_CORRECT_ESCALATE: float = 0.30
REWARD_HIGH_SATISFACTION: float = 0.20
REWARD_FAST_RESOLUTION: float = 0.20
PENALTY_UNNECESSARY_ESCALATE: float = -0.30
PENALTY_REPEAT_ACTION: float = -0.30

MAX_STEPS: int = 5

VALID_CATEGORIES = frozenset([
    "login_issue", "payment_failure", "refund_request", "delivery_delay"
])

_HELPFUL_KWS = frozenset([
    "refund", "help", "solution", "resolve", "assist", "fix",
    "support", "understand", "process", "issue", "address",
])
_POLITE_KWS = frozenset([
    "sorry", "apologize", "apologies", "thank", "appreciate", "please",
])

# ── Dynamic Ticket Pool (5 categories x 3 tones) ──────────────────────────────
_TICKET_POOL: List[Dict[str, str]] = [
    {
        "category": "login_issue", "urgency": "medium", "tone": "angry",
        "message": "I CANNOT log into my account! Fix this NOW!"
    },
    {
        "category": "login_issue", "urgency": "low", "tone": "polite",
        "message": "Hello, I'm having trouble logging in. Could you help me? Thank you!"
    },
    {
        "category": "login_issue", "urgency": "medium", "tone": "confused",
        "message": "Hi, I'm not sure what's happening but I can't get into my account?"
    },
    {
        "category": "payment_failure", "urgency": "high", "tone": "angry",
        "message": "My payment of $299 was CHARGED TWICE! I demand an immediate refund!"
    },
    {
        "category": "payment_failure", "urgency": "medium", "tone": "polite",
        "message": "Good morning, my payment didn't go through. Happy to provide info."
    },
    {
        "category": "payment_failure", "urgency": "low", "tone": "confused",
        "message": "Um, I tried to pay but it said declined. Is it something I'm doing wrong?"
    },
    {
        "category": "refund_request", "urgency": "medium", "tone": "angry",
        "message": "I want my money back RIGHT NOW. False advertising!"
    },
    {
        "category": "refund_request", "urgency": "low", "tone": "polite",
        "message": "Hi there, I'd like to request a refund for my order #88231. Thank you!"
    },
    {
        "category": "refund_request", "urgency": "low", "tone": "confused",
        "message": "Does 30-day returns mean I can get a refund? How do I start?"
    },
    {
        "category": "delivery_delay", "urgency": "medium", "tone": "angry",
        "message": "My package was supposed to arrive 5 days ago! I want answers NOW!"
    },
    {
        "category": "delivery_delay", "urgency": "low", "tone": "polite",
        "message": "Hello, wanted to follow up on my delivery. Tracking hasn't updated."
    },
    {
        "category": "delivery_delay", "urgency": "low", "tone": "confused",
        "message": "My tracking says delivered but I didn't get it. What to do?"
    },
]

def _generate_ticket_id(rng: random.Random) -> str:
    return f"TKT-2024{rng.randint(1000, 9999)}"


class SupportEnvironment(Environment[Action, Observation, State]):
    """Hybrid Customer Support and Business Environment."""

    def __init__(self) -> None:
        super().__init__()
        self.max_steps: int = MAX_STEPS
        self.state_obj: State = State()
        self._rng: random.Random = random.Random()
        # Keep track of global metrics across episodes for continuity, if desired
        self.global_tickets_resolved = 0

    def reset(self, seed: Optional[int] = None, **kwargs) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        template = self._rng.choice(_TICKET_POOL)
        
        # Start customer satisfaction based on their initial tone
        initial_sat = 0.5
        if template["tone"] == "angry":
            initial_sat = 0.2
        elif template["tone"] == "polite":
            initial_sat = 0.8
            
        self.state_obj = State(
            ticket_id=_generate_ticket_id(self._rng),
            customer_message=template["message"],
            true_category=template["category"],
            category=None,
            conversation_history=[f"[Customer] {template['message']}"],
            urgency=template["urgency"],
            resolved=False,
            step_count=0,
            has_categorized=False,
            has_replied=False,
            has_escalated=False,
            last_action_type=None,
            action_counts={},
            episode_history=[],
            satisfaction_score=initial_sat,
            cost_spent=0.0,
            tickets_resolved=self.global_tickets_resolved
        )

        obs = self.state_obj.to_observation()
        obs.reward = 0.0
        obs.done = False
        obs.info = {}
        return obs

    def step(self, action: Action) -> Observation:
        s = self.state_obj
        s.step_count += 1
        s.last_action_type = action.action_type
        
        s.action_counts[action.action_type] = s.action_counts.get(action.action_type, 0) + 1
        count = s.action_counts[action.action_type]

        s.conversation_history.append(f"[{action.action_type.upper()}] {action.content}")

        # ── Apply action and calculate reward ─────────────────────────────────
        total_reward, breakdown = self._calculate_reward_and_update_metrics(action, s, count)

        # ── Resolution logic ──────────────────────────────────────────────────
        if s.urgency == "high":
            s.resolved = s.has_categorized and s.has_replied and s.has_escalated
        else:
            s.resolved = s.has_categorized and s.has_replied

        if s.resolved:
            self.global_tickets_resolved += 1
            s.tickets_resolved = self.global_tickets_resolved
            # Fast resolution bonus
            if s.step_count <= 2:
                total_reward += REWARD_FAST_RESOLUTION
                breakdown["fast_resolution"] = REWARD_FAST_RESOLUTION

        terminated = s.resolved
        truncated = s.step_count >= self.max_steps
        done = terminated or truncated
        
        # Clamp total reward
        total_reward = float(max(0.0, min(1.0, total_reward)))

        # ── Record step in episode history (for graders) ──────────────────────
        step_record: Dict[str, Any] = {
            "step": s.step_count,
            "action_type": action.action_type,
            "content": action.content,
            "true_category": s.true_category,
            "agent_category": s.category,
            "urgency": s.urgency,
            "resolved": s.resolved,
            "reward": total_reward,
            "satisfaction_score": s.satisfaction_score,
            "cost_spent": s.cost_spent,
            "breakdown": breakdown,
        }
        s.episode_history.append(step_record)

        info: Dict[str, Any] = {
            "progress": round(s.step_count / self.max_steps, 2),
            "step": s.step_count,
            "detected_category": s.category,
            "true_category": s.true_category,
            "resolved": s.resolved,
            "urgency": s.urgency,
            "satisfaction_score": s.satisfaction_score,
            "cost_spent": s.cost_spent,
            "reward_breakdown": breakdown,
        }

        obs = s.to_observation()
        obs.reward = total_reward
        obs.done = done
        obs.info = info
        return obs

    def state(self) -> Dict[str, Any]:
        s = self.state_obj
        return {
            "ticket_id": s.ticket_id,
            "customer_message": s.customer_message,
            "true_category": s.true_category,
            "category": s.category,
            "conversation_history": list(s.conversation_history),
            "urgency": s.urgency,
            "resolved": s.resolved,
            "step_count": s.step_count,
            "satisfaction_score": s.satisfaction_score,
            "cost_spent": s.cost_spent,
            "tickets_resolved": s.tickets_resolved,
            "action_counts": dict(s.action_counts),
        }

    def typed_state(self) -> State:
        return self.state_obj

    def _calculate_reward_and_update_metrics(self, action: Action, s: State, count: int) -> tuple[float, Dict[str, float]]:
        breakdown: Dict[str, float] = {}
        total: float = 0.0

        if count > 1:
            breakdown["repeated_action"] = PENALTY_REPEAT_ACTION
            total += PENALTY_REPEAT_ACTION
            s.satisfaction_score = max(0.0, s.satisfaction_score - 0.1)

        if action.action_type == "categorize":
            s.has_categorized = True
            agent_cat = action.content.strip().lower()
            s.category = agent_cat
            if agent_cat == s.true_category:
                breakdown["correct_categorize"] = REWARD_CORRECT_CATEGORIZE
                total += REWARD_CORRECT_CATEGORIZE
            elif agent_cat not in VALID_CATEGORIES:
                breakdown["irrelevant_action"] = PENALTY_REPEAT_ACTION
                total += PENALTY_REPEAT_ACTION

        elif action.action_type == "reply":
            s.has_replied = True
            content_lower = action.content.lower()
            is_helpful = any(kw in content_lower for kw in _HELPFUL_KWS)
            is_polite = any(kw in content_lower for kw in _POLITE_KWS)
            
            if is_helpful:
                breakdown["helpful_response"] = REWARD_HELPFUL_REPLY
                total += REWARD_HELPFUL_REPLY
                s.satisfaction_score = min(1.0, s.satisfaction_score + 0.3)
            else:
                breakdown["irrelevant_action"] = PENALTY_REPEAT_ACTION
                total += PENALTY_REPEAT_ACTION
                s.satisfaction_score = max(0.0, s.satisfaction_score - 0.2)
                
            if is_polite:
                s.satisfaction_score = min(1.0, s.satisfaction_score + 0.1)

            if s.satisfaction_score > 0.8:
                breakdown["high_satisfaction"] = REWARD_HIGH_SATISFACTION
                total += REWARD_HIGH_SATISFACTION

        elif action.action_type == "escalate":
            s.has_escalated = True
            s.cost_spent += 50.0  # Business cost incurred
            
            if s.urgency == "high":
                breakdown["correct_escalate"] = REWARD_CORRECT_ESCALATE
                total += REWARD_CORRECT_ESCALATE
            else:
                breakdown["unnecessary_escalation"] = PENALTY_UNNECESSARY_ESCALATE
                total += PENALTY_UNNECESSARY_ESCALATE
                s.satisfaction_score = max(0.0, s.satisfaction_score - 0.1)

        return total, breakdown
