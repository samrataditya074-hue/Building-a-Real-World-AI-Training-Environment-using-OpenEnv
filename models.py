"""
models.py — Data blueprints for the Autonomous CEO AI Simulator.

Plain-English overview:
  - Employee : one person working at the company
  - Action   : the 8 decisions the AI CEO makes each quarter
  - Observation : what the AI "sees" — 14 normalised numbers describing company health
  - Reward   : the score the AI receives after each decision
  - State    : the complete, uncompressed truth about the company (used internally)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Employee
# Represents one real person working at the company.
# Each quarter their performance and morale shift slightly based on what the AI did.
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Employee:
    """One employee at the company."""

    id: int           # Unique badge number
    name: str         # Random first name (for the live roster display)
    department: str   # Which team: Finance, Sales, HR, Customer, or Ops
    performance: float  # How well they work (0–100). Drifts with morale.
    salary: float       # Monthly pay in dollars ($1,500 – $10,000)
    morale: float       # How happy they are (0–100). Falls if fired friends or pay-cut.
    tenure: int         # How many quarters they've worked here. Grows each step.


# ──────────────────────────────────────────────────────────────────────────────
# Action (Pydantic model — validates ranges automatically)
# These are the 8 levers the AI pushes or pulls each quarter.
# Every value must be between -1.0 and +1.0.
# ──────────────────────────────────────────────────────────────────────────────
class Action(BaseModel):
    """
    The CEO's 8 strategic decisions for one quarter.
    All values are continuous in [-1.0, +1.0].
    The environment translates these into concrete business changes.
    """

    price_adjustment: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Adjust product price vs competitor. "
            "+1 = raise $8 (higher margin, fewer buyers). "
            "-1 = lower $8 (more buyers, thinner margin)."
        )
    )
    marketing_push: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "Marketing spend this quarter. "
            "0 = $0. +1 = $600 → boosts customer demand and satisfaction."
        )
    )
    hire_fire: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Workforce change. "
            "+1 = hire 5 people. -1 = fire 5 lowest performers "
            "(firing hurts remaining morale by 3% per person)."
        )
    )
    rd_investment: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description=(
            "R&D budget this quarter. "
            "0 = $0. +1 = $400. "
            "Increments R&D progress (0–100); unlocks efficiency and reputation bonuses."
        )
    )
    salary_adjustment: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Salary change. "
            "+1 = +10% merit bonuses to top performers (morale +15). "
            "-1 = -10% company-wide cut (saves cash but morale -8 each)."
        )
    )
    task_allocation: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Team structure. "
            "+1 = cross-train teams (flexible, efficiency +2). "
            "-1 = specialise teams (efficiency +3.5, individual performance +1.5)."
        )
    )
    crisis_response: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Crisis handling (only active when ≥1 crisis flag is set). "
            "+1 = invest through crisis (-$500, +5% morale, +3 reputation). "
            "-1 = emergency cost-cut (+$300, -5% morale)."
        )
    )
    budget_shift: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Reallocate budget between Sales and Operations. "
            "+1 = aggressive growth (more sales spend). "
            "-1 = conservative savings (more ops spend)."
        )
    )

    # Allow construction from a plain dict or numpy array for backward-compat
    model_config = {"arbitrary_types_allowed": True}


# ──────────────────────────────────────────────────────────────────────────────
# Observation (Pydantic model — what the AI sees each quarter)
# 14 normalised numbers. All big raw values are divided by constants so the
# AI's neural network can process them easily (numbers near 0–1 work best).
# ──────────────────────────────────────────────────────────────────────────────
class Observation(BaseModel):
    """
    The agent's view of the company each quarter — 14 normalised floats.
    Raw values are divided by normalization constants to keep them near 0–1.
    """

    # ── Financial health ──────────────────────────────────────────────────────
    cash_norm: float = Field(
        description="Cash ÷ $200,000. 1.0 = $200k (starting level). >1 = very wealthy."
    )
    revenue_norm: float = Field(
        description="Quarterly revenue ÷ $5,000."
    )

    # ── People metrics ────────────────────────────────────────────────────────
    customer_satisfaction_norm: float = Field(
        description="Customer happiness (0–100) ÷ 100."
    )
    employee_morale_norm: float = Field(
        description="Average employee happiness (0–100) ÷ 100. Below 0.35 = morale crisis."
    )

    # ── Operations ────────────────────────────────────────────────────────────
    inventory_norm: float = Field(
        description="Units in stock ÷ 1,000. Below 0.05 = inventory crisis."
    )
    market_trend: float = Field(
        description="Economy health multiplier. 1.0 = neutral. >1 = boom. <1 = recession."
    )
    total_employees_norm: float = Field(
        description="Number of employees ÷ 50. 0.4 = 20 employees (starting level)."
    )

    # ── Brand & technology ────────────────────────────────────────────────────
    brand_reputation_norm: float = Field(
        description="Brand trust score (0–100) ÷ 100. Below 0.25 = reputation crisis."
    )
    operational_efficiency_norm: float = Field(
        description="Operational efficiency (0–100) ÷ 100. Boosted by R&D and specialisation."
    )
    rd_progress_norm: float = Field(
        description="R&D advancement level (0–100) ÷ 100. Unlocks bonuses at 0.3 and 0.7."
    )

    # ── Risk signals ──────────────────────────────────────────────────────────
    debt_norm: float = Field(
        description="Total debt ÷ $10,000."
    )
    cash_crisis_flag: float = Field(
        description="1.0 if cash < $2,000 (near-bankruptcy). 0.0 otherwise."
    )
    morale_crisis_flag: float = Field(
        description="1.0 if morale < 35% (people starting to quit). 0.0 otherwise."
    )
    competitor_price_norm: float = Field(
        description="Competitor's current price ÷ $100."
    )

    # ── OpenEnv standard fields (added by step() / reset()) ──────────────────
    reward: float = Field(default=0.0, description="Reward received this step.")
    done: bool = Field(default=False, description="True if the episode has ended.")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra debug info: thought, actions list, crisis count, reward breakdown."
    )

    model_config = {"arbitrary_types_allowed": True}

    def to_array(self) -> np.ndarray:
        """
        Convert observation to a flat numpy array for RL model consumption.
        Order matches openenv.yaml observation_space fields (indices 0–13).
        """
        return np.array([
            self.cash_norm,
            self.revenue_norm,
            self.customer_satisfaction_norm,
            self.employee_morale_norm,
            self.inventory_norm,
            self.market_trend,
            self.total_employees_norm,
            self.brand_reputation_norm,
            self.operational_efficiency_norm,
            self.rd_progress_norm,
            self.debt_norm,
            self.cash_crisis_flag,
            self.morale_crisis_flag,
            self.competitor_price_norm,
        ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Reward — structured breakdown of the per-step score
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Reward:
    """
    Structured breakdown of the reward signal each quarter.

    The AI tries to maximise `total` over an entire episode.
    Positive components reward good business outcomes.
    Negative components penalise harmful decisions.
    """

    total: float         # Net reward = pos - neg + shaping terms
    pos: float           # Positive contribution (profit, morale, reputation, R&D)
    neg: float           # Negative contribution (losses, crises, low morale/rep)

    breakdown: Dict[str, float] = field(default_factory=dict)
    # breakdown keys (example):
    #   "profit_component"   → contribution from profit/loss
    #   "morale_component"   → contribution from employee morale
    #   "rep_component"      → contribution from brand reputation
    #   "rd_component"       → contribution from R&D progress
    #   "profit_delta"       → per-step profit improvement shaping bonus
    #   "morale_delta"       → per-step morale improvement shaping bonus
    #   "rd_payoff"          → cumulative R&D payoff shaping
    #   "fire_penalty"       → penalty for firing many people at once


# ──────────────────────────────────────────────────────────────────────────────
# State — the complete truth about the company (used internally by environment.py)
# This is NOT what the AI sees directly. It is converted to Observation via to_observation().
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class State:
    """
    The entire underlying business reality.

    Think of this as the full company database that the simulation uses.
    The Observation above is a simplified, normalised window into this data
    that the AI actually reads.
    """

    # ── Core financials ───────────────────────────────────────────────────────
    # Starting cash: $200,000. If this hits $0 — company goes bankrupt.
    cash: float = 200_000.0
    revenue: float = 0.0    # Money earned from product sales this quarter
    profit: float = 0.0     # Revenue minus all costs (payroll, marketing, R&D, fixed)
    debt: float = 0.0       # Outstanding debt (not actively used in v2 economy)

    # ── People & culture ──────────────────────────────────────────────────────
    # customer_satisfaction: how happy our buyers are (0–100). Falls if price goes up.
    customer_satisfaction: float = 75.0
    # employee_morale: average happiness of all staff (0–100). Drops when people are fired.
    employee_morale: float = 80.0

    # ── Operations ────────────────────────────────────────────────────────────
    inventory: float = 500.0       # Units of product in stock ready to sell
    market_trend: float = 1.0      # External economy health (random walk, 0.4–1.6)
    competitor_price: float = 50.0 # What the rival charges. AI can undercut or match.

    # ── Company capabilities ──────────────────────────────────────────────────
    total_employees: int = 20
    avg_salary: float = 3000.0
    # rd_progress: how advanced our technology is (0–100).
    # Think of it like a technology level — starts at 0, grows with R&D investment.
    rd_progress: float = 0.0
    # brand_reputation: how much customers and the market trust us (0–100).
    brand_reputation: float = 60.0
    # operational_efficiency: how smoothly we run day-to-day operations (0–100).
    operational_efficiency: float = 70.0

    # ── Department budget allocations (fraction of total, sum ≈ 1.0) ─────────
    finance_budget: float = 0.20
    sales_budget: float = 0.25
    hr_budget: float = 0.15
    customer_budget: float = 0.15
    ops_budget: float = 0.25

    # ── Crisis flags (True = emergency, triggers crisis_response lever) ───────
    cash_crisis: bool = False        # Cash < $2,000
    morale_crisis: bool = False      # Morale < 35%
    inventory_crisis: bool = False   # Inventory < 50 units
    reputation_crisis: bool = False  # Brand reputation < 25

    # ── Narrative / history ───────────────────────────────────────────────────
    news: str = "Company founded. Market stable."
    quarter: int = 0
    headline: str = "Welcome to your first Quarter as CEO!"

    employees: List[Employee] = field(default_factory=list)
    last_actions: Dict[str, Any] = field(default_factory=dict)
    event_history: List[str] = field(default_factory=list)
    competitor_price_history: List[float] = field(default_factory=list)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    merger_progress: float = 0.0  # Track progress for negotiation tasks


    # ── Computed helpers ──────────────────────────────────────────────────────

    def dept_scores(self) -> Dict[str, float]:
        """
        Department performance scores (0–10) — used by the 3D bar chart.
        Each bar height = how healthy that department is right now.
        """
        return {
            "Finance": min(10.0, self.cash / 2_000.0),
            "Sales": min(10.0, self.revenue / 1_000.0),
            "Customer": self.customer_satisfaction / 10.0,
            "HR": self.employee_morale / 10.0,
            "Operations": min(10.0, self.operational_efficiency / 10.0),
        }

    def get_valuation(self) -> float:
        """
        Company's total intrinsic value — used for the growth chart.

        Formula: Cash + (Brand × $100) + (Morale × $10) + (R&D × $50)
        A thriving company is worth more than just its bank balance —
        happy employees and strong tech add real value.
        """
        return (
            self.cash
            + self.brand_reputation * 100.0
            + self.employee_morale * 10.0
            + self.rd_progress * 50.0
        )

    def get_roster(self) -> List[List[Any]]:
        """Returns employee roster sorted by performance (best first) for the dashboard table."""
        roster = []
        for e in sorted(self.employees, key=lambda x: x.performance, reverse=True):
            roster.append([
                f"EMP-{e.id:03d}",
                e.name,
                e.department,
                f"${e.salary:.0f}",
                f"{e.performance:.1f}",
                f"{e.morale:.1f}",
                e.tenure,
            ])
        return roster

    def to_observation(self) -> "Observation":
        """
        Converts the full State into the normalised Observation the AI reads.

        Each raw value is divided by a normalization constant so the AI's
        neural network receives numbers near 0–1 (much easier to learn from).
        """
        return Observation(
            cash_norm=self.cash / 200_000.0,
            revenue_norm=self.revenue / 5_000.0,
            customer_satisfaction_norm=self.customer_satisfaction / 100.0,
            employee_morale_norm=self.employee_morale / 100.0,
            inventory_norm=self.inventory / 1_000.0,
            market_trend=self.market_trend,
            total_employees_norm=self.total_employees / 50.0,
            brand_reputation_norm=self.brand_reputation / 100.0,
            operational_efficiency_norm=self.operational_efficiency / 100.0,
            rd_progress_norm=self.rd_progress / 100.0,
            debt_norm=self.debt / 10_000.0,
            cash_crisis_flag=1.0 if self.cash_crisis else 0.0,
            morale_crisis_flag=1.0 if self.morale_crisis else 0.0,
            competitor_price_norm=self.competitor_price / 100.0,
        )
