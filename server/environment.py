"""
server/environment.py — Main CEO simulation engine (OpenEnv-compliant).

Plain-English overview:
  This file is the heart of the simulation. Every quarter, it:
    1. Receives 8 decisions from the AI CEO (the Action object)
    2. Runs the business world forward through 15 realistic stages
       (market changes, hiring, payroll, sales, R&D, reputation, crises …)
    3. Returns what happened (the Observation) and a reward score

  The reward tells the AI how well it did. Over millions of practice rounds,
  the AI gradually learns which decisions lead to the best outcomes.

OpenEnv API contract:
  reset(seed=None)  → Observation
  step(action)      → Observation  (obs.reward, obs.done, obs.info set)
  state()           → dict         (OpenEnv spec: always returns dict)
  typed_state()     → State        (internal typed access)
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
from openenv.core.env_server import Environment

from models import Action, Observation, Reward, State, Employee
from graders import CEORubric

# ── Logging setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("ceo_env")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [CEO-ENV] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)

# Default to ERROR to prevent console flooding during massive RL training loops.
env_log_level = os.getenv("ENV_LOG_LEVEL", "ERROR").upper()
logger.setLevel(getattr(logging, env_log_level, logging.ERROR))

# ── Reward normalisation constants (documented here for openenv.yaml reference) ──
# These constants convert large dollar amounts into small numbers near 0–1
# so the reward signal is stable and easy for the neural network to learn from.
REWARD_NORM_PROFIT: float = 80.0          # Divides profit/loss contribution
REWARD_NORM_SATISFACTION: float = 40.0    # Divides customer satisfaction contribution
REWARD_NORM_MORALE: float = 60.0          # Divides morale contribution
REWARD_NORM_REPUTATION: float = 80.0      # Divides brand reputation contribution
REWARD_NORM_PROFIT_DELTA: float = 5_000.0 # Normalises quarter-over-quarter profit change
REWARD_NORM_MORALE_DELTA: float = 10.0    # Normalises quarter-over-quarter morale change
REWARD_RD_SCALE: float = 0.20             # Maximum R&D payoff contribution per step
REWARD_FIRE_PENALTY: float = 0.30         # Penalty applied per "fraction of team fired"
REWARD_TERMINAL_PENALTY: float = 50.0     # One-time penalty for bankruptcy / collapse

# Employee name pool for random hiring
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Blake", "Drew", "Sage", "Reese", "Dakota", "Finley", "Rowan", "Skyler",
    "Charlie", "Emerson", "Hayden", "Parker", "Jamie", "Peyton", "Kai", "Tatum",
]
DEPARTMENTS = ["Finance", "Sales", "HR", "Customer", "Ops"]


def _make_employee(eid: int, rng: random.Random) -> Employee:
    """Create a new employee with randomised stats."""
    return Employee(
        id=eid,
        name=rng.choice(FIRST_NAMES),
        department=rng.choice(DEPARTMENTS),
        performance=rng.uniform(40.0, 95.0),
        salary=rng.uniform(2_000.0, 5_000.0),
        morale=rng.uniform(50.0, 95.0),
        tenure=0,
    )


class CEOEnvironment(Environment[Action, Observation, State]):
    """
    OpenEnv-compliant Autonomous CEO AI Simulator.

    State space : 14-dimensional normalised Observation
    Action space: 8-dimensional continuous Action (all values in [-1, +1])
    Reward      : per-step continuous signal (see REWARD_NORM_* constants above)

    The simulation runs for up to max_steps quarters. An episode ends early
    if the company goes bankrupt, loses all customers, or drops below 3 staff.
    """

    def __init__(self) -> None:
        super().__init__()
        self.max_steps: int = 200             # Maximum quarters per episode
        self.state_obj: State = State()       # Full business state
        self.neg_reward: float = 0.0          # Negative reward component this step
        
        # OpenEnv Rubric - provides formal grading
        self.task_id = "grow_val_medium"
        self.rubric = CEORubric(self.task_id)
        self.action_labels: List[str] = []    # Human-readable action descriptions

        # Previous-step values for delta-based reward shaping
        self._prev_profit: float = 0.0
        self._prev_morale: float = 80.0
        self._fired_this_step: int = 0        # How many people were fired this step

        # RNG (seeded in reset())
        self._rng: random.Random = random.Random()
        self._np_rng: np.random.Generator = np.random.default_rng()

        # Episode trace (written to disk when TRACE_LOGGING=1 env var is set)
        self.episode_trace: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # reset() — Start a new episode (new company, fresh state)
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, **kwargs) -> Observation:
        """
        Reset the simulation to a fresh state.

        Args:
            seed: If provided, seeds all random number generators so the
                  episode is fully reproducible. Same seed = same events.

        Returns:
            Initial Observation (all at starting values).
        """
        # Seed all RNGs for determinism
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)
            logger.info("Episode reset with seed=%d", seed)
        else:
            self._rng = random.Random()
            self._np_rng = np.random.default_rng()
            logger.info("Episode reset (no seed)")

        # Create 20 employees with random attributes
        employees = [_make_employee(i, self._rng) for i in range(20)]
        self.state_obj = State(employees=employees)
        self.state_obj.competitor_price_history.append(self.state_obj.competitor_price)

        # Reset tracking variables
        self.action_labels = []
        self._prev_profit = 0.0
        self._prev_morale = self.state_obj.employee_morale
        self._fired_this_step = 0
        self.episode_trace = []

        obs = self.state_obj.to_observation()
        obs.reward = 0.0
        obs.done = False
        obs.info = {}

        # Reset the rubric state
        self.task_id = kwargs.get("task_id", self.task_id)
        self.rubric = CEORubric(self.task_id)
        self._reset_rubric()

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # step() — Advance one quarter with the given action
    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: Action) -> Observation:
        """
        Execute one business quarter.

        The simulation runs through 15 stages in order:
          market → HR → salary → task alloc → employee dynamics →
          budget → demand → operations → R&D → financials →
          reputation → crises → crisis response → reward → termination

        Args:
            action: The CEO's 8 decisions for this quarter.

        Returns:
            Observation with reward, done flag, and info dict attached.
        """
        s = self.state_obj
        s.quarter += 1
        self.action_labels = []
        self._fired_this_step = 0

        # ── Decode the 8 action levers into real-world magnitudes ─────────────
        # Each lever (-1 to +1) is scaled to a meaningful business unit:
        price_adj = action.price_adjustment * 8.0          # dollars change in price
        mkt_spend = max(0.0, action.marketing_push * 600.0) # dollars spent on marketing
        hire_fire = int(round(action.hire_fire * 5))         # people hired (>0) or fired (<0)
        rd_invest = max(0.0, action.rd_investment * 400.0)   # dollars invested in R&D
        salary_adj = action.salary_adjustment * 0.10         # ±10% salary change
        task_alloc = action.task_allocation
        crisis_resp = action.crisis_response
        budget_shift = action.budget_shift

        logger.info(
            "Q%d | price=%.1f mkt=%.0f hire=%+d rd=%.0f sal=%.0f%% budget=%.2f",
            s.quarter, price_adj, mkt_spend, hire_fire, rd_invest,
            salary_adj * 100, budget_shift
        )

        # ── Stage 1: Market Dynamics & Random World Events ────────────────────
        # The economy changes randomly each quarter — just like the real world.
        # Small drift up or down, plus rare big events (1.5% each).
        market_shift = self._np_rng.normal(0, 0.06)
        s.market_trend = float(np.clip(s.market_trend + market_shift, 0.4, 1.6))

        event_msg: Optional[str] = None
        shock = self._rng.random()

        if shock < 0.015:
            # Global recession — worst possible event
            s.market_trend *= 0.5
            s.customer_satisfaction -= 15
            event_msg = "📉 GLOBAL RECESSION! Markets plunge, consumer spending halted."
        elif shock > 0.985:
            # Economic boom — best possible event
            s.market_trend *= 1.4
            s.brand_reputation = min(100.0, s.brand_reputation + 20)
            event_msg = "🚀 ECONOMIC BOOM! Golden quarter for the industry!"
        elif 0.015 <= shock <= 0.030:
            # Labor strike — operations grind to a halt
            s.inventory = max(0.0, s.inventory - 500)
            s.operational_efficiency = max(20.0, s.operational_efficiency - 20)
            event_msg = "⛓️ LABOR STRIKE! Operations halted, massive supply chain shock."
        elif 0.970 <= shock <= 0.985:
            # R&D breakthrough — massive technology leap
            s.rd_progress = min(100.0, s.rd_progress + 25)
            s.operational_efficiency = min(100.0, s.operational_efficiency + 15)
            event_msg = "🧬 TECH BREAKTHROUGH! R&D discovers revolutionary workflow."
        elif 0.030 <= shock <= 0.045:
            # Competitor scandal — their customers flee to us
            s.competitor_price *= 1.3
            s.market_trend *= 1.1
            event_msg = "🏛️ COMPETITOR SCANDAL! Rival investigated, customers flock to us!"

        if event_msg:
            s.news = event_msg
            s.event_history.append(f"Q{s.quarter}: {event_msg}")
            logger.info("Event: %s", event_msg)

        # ── Stage 2: HR — Hire or Fire ────────────────────────────────────────
        # Hiring grows the team and production capacity.
        # Firing targets lowest performers first, but damages morale.
        if hire_fire > 0:
            hired_names: List[str] = []
            for _ in range(hire_fire):
                new_emp = _make_employee(len(s.employees), self._rng)
                s.employees.append(new_emp)
                hired_names.append(new_emp.name)
            self.action_labels.append(f"Hired {hire_fire}: {', '.join(hired_names)}")
            logger.info("Hired %d employees: %s", hire_fire, ', '.join(hired_names))

        elif hire_fire < 0:
            # Fire the lowest performers (most humane way to downsize)
            fires = min(abs(hire_fire), len(s.employees))
            if fires > 0:
                worst = sorted(s.employees, key=lambda e: e.performance)[:fires]
                fired_names: List[str] = []
                for w in worst:
                    s.employees.remove(w)
                    fired_names.append(f"{w.name} ({w.department})")
                self.action_labels.append(f"Fired lowest performers: {', '.join(fired_names)}")
                # Firing hurts the morale of remaining employees
                s.employee_morale -= fires * 3.0
                self._fired_this_step = fires
                logger.info("Fired %d employees. Morale impact: -%.0f%%", fires, fires * 3.0)

        s.total_employees = len(s.employees)

        # ── Stage 3: Salary Adjustments ───────────────────────────────────────
        # Raises go to top performers only (merit-based).
        # Cuts apply company-wide and always hurt morale.
        if abs(salary_adj) > 0.01:
            bonused_names: List[str] = []
            for emp in s.employees:
                if salary_adj > 0 and emp.performance > 75:
                    # Double the raise for high performers — reward excellence
                    emp.salary *= (1.0 + salary_adj * 2)
                    emp.morale += 15.0
                    bonused_names.append(emp.name)
                elif salary_adj < 0:
                    emp.salary *= (1.0 + salary_adj)
                    emp.morale -= 8.0
                emp.salary = max(1_500.0, min(emp.salary, 10_000.0))

            if salary_adj > 0 and bonused_names:
                self.action_labels.append(
                    f"Performance Bonuses to top {len(bonused_names)}: "
                    f"{', '.join(bonused_names[:3])}{'...' if len(bonused_names) > 3 else ''}"
                )
            elif salary_adj < 0:
                self.action_labels.append(
                    f"Company-wide Salary Cut: {abs(salary_adj) * 100:.0f}%"
                )

        # ── Stage 4: Task / Team Allocation ──────────────────────────────────
        # Cross-training builds flexibility; specialisation maximises efficiency.
        if task_alloc > 0.3:
            s.operational_efficiency = min(100.0, s.operational_efficiency + 2.0)
            self.action_labels.append("Cross-trained teams for flexibility")
        elif task_alloc < -0.3:
            s.operational_efficiency = min(100.0, s.operational_efficiency + 3.5)
            for emp in s.employees:
                emp.performance = min(100.0, emp.performance + 1.5)
            self.action_labels.append("Specialised departments for focus")

        # ── Stage 5: Employee Natural Dynamics ────────────────────────────────
        # Every quarter: employees' performance and morale drift slightly.
        # Happy employees (high morale) tend to improve; miserable ones decline.
        for emp in s.employees:
            emp.tenure += 1
            # Performance shifts: random noise + morale influence
            perf_delta = self._rng.uniform(-2.0, 2.0) + (emp.morale - 50.0) * 0.02
            emp.performance = float(np.clip(emp.performance + perf_delta, 10.0, 100.0))
            # Morale shifts: random noise + salary policy effect
            morale_delta = self._rng.uniform(-3.0, 3.0) + salary_adj * 10.0
            emp.morale = float(np.clip(emp.morale + morale_delta, 5.0, 100.0))

        if s.employees:
            s.employee_morale = float(np.mean([e.morale for e in s.employees]))
            s.avg_salary = float(np.mean([e.salary for e in s.employees]))

        # ── Stage 6: Budget Allocation ────────────────────────────────────────
        # Shifts money flow between Sales (growth) and Operations (efficiency).
        shift = budget_shift * 0.03
        s.sales_budget = float(np.clip(s.sales_budget + shift, 0.05, 0.50))
        s.ops_budget = float(np.clip(s.ops_budget - shift * 0.5, 0.05, 0.50))
        if abs(budget_shift) > 0.3:
            mode = "aggressive growth" if budget_shift > 0 else "conservative savings"
            self.action_labels.append(f"Budget shifted to {mode}")

        # ── Stage 7: Customer Demand Calculation ──────────────────────────────
        # How many customers want to buy this quarter?
        # Demand grows with: big team, good market, cheap prices, heavy marketing, strong brand.
        our_price = s.competitor_price + price_adj
        # Competitor slowly reacts to our pricing (they won't let us undercut forever)
        competitor_reaction = self._np_rng.normal(0, 1.0)
        s.competitor_price = s.competitor_price * 0.8 + our_price * 0.2 + competitor_reaction
        s.competitor_price = float(np.clip(s.competitor_price, 20.0, 150.0))
        s.competitor_price_history.append(s.competitor_price)

        # Base demand: bigger team + stronger economy = more potential customers
        base_demand = 450.0 * s.market_trend * (s.total_employees / 20.0)
        # Price factor: competing on price brings in more customers
        price_factor = max(0.2, 2.0 - (our_price / max(1.0, s.competitor_price)))
        # Marketing bonus: $10 of marketing ≈ 1 extra customer
        mkt_bonus = mkt_spend / 10.0
        # Reputation bonus: trusted brands attract more buyers automatically
        rep_bonus = s.brand_reputation * 1.5
        demand = max(0.0, base_demand * price_factor + mkt_bonus + rep_bonus)

        if mkt_spend > 200:
            self.action_labels.append(f"Marketing push: ${mkt_spend:.0f}")

        # ── Stage 8: Operations — Sell Products, Replenish Stock ──────────────
        sales = min(s.inventory, demand)   # Can't sell more than we have in stock
        s.inventory -= sales
        # Production: bigger team + efficient operations → more goods produced
        production = (s.total_employees * 25.0) + (s.operational_efficiency * 3.0)
        s.inventory += production
        s.inventory = max(0.0, s.inventory)
        s.revenue = sales * our_price

        if price_adj != 0:
            direction = "↑" if price_adj > 0 else "↓"
            self.action_labels.append(f"Price {direction} ${abs(price_adj):.1f}")

        # ── Stage 9: R&D Progress ─────────────────────────────────────────────
        # R&D is like planting trees — costs now, but improves everything over time.
        if rd_invest > 50:
            s.rd_progress = min(100.0, s.rd_progress + rd_invest / 100.0)
            self.action_labels.append(f"R&D investment: ${rd_invest:.0f}")
        # Long-term R&D benefits (unlock at 30% and 70% progress)
        if s.rd_progress > 30:
            s.operational_efficiency = min(100.0, s.operational_efficiency + 0.5)
        if s.rd_progress > 70:
            s.brand_reputation = min(100.0, s.brand_reputation + 0.3)

        # ── Stage 10: Financials ──────────────────────────────────────────────
        # Revenue - all costs = Profit. Profit goes into (or drains from) cash reserves.
        quarterly_payroll = s.avg_salary * s.total_employees / 4.0  # paid each quarter
        total_costs = 400.0 + mkt_spend + rd_invest + quarterly_payroll
        s.profit = s.revenue - total_costs
        s.cash += s.profit

        logger.info(
            "Q%d Financials | Revenue=%.0f Costs=%.0f Profit=%.0f Cash=%.0f",
            s.quarter, s.revenue, total_costs, s.profit, s.cash
        )

        # ── Stage 11: Customer Satisfaction & Brand Reputation ────────────────
        # Satisfaction: marketing helps, price hikes hurt, operational efficiency helps.
        s.customer_satisfaction += (
            (mkt_spend / 800.0)
            - (price_adj / 15.0)
            + (s.operational_efficiency - 70.0) * 0.02
        )
        s.customer_satisfaction = float(np.clip(s.customer_satisfaction, 0.0, 100.0))
        # Brand reputation: profitable companies build a better brand over time.
        s.brand_reputation += s.profit / 5_000.0
        s.brand_reputation = float(np.clip(s.brand_reputation, 0.0, 100.0))
        s.operational_efficiency = float(np.clip(s.operational_efficiency, 20.0, 100.0))

        # ── Stage 12: Crisis Detection ────────────────────────────────────────
        # Four warning lights. If any are on, the crisis_response lever becomes active.
        s.cash_crisis = s.cash < 2_000.0
        s.morale_crisis = s.employee_morale < 35.0
        s.inventory_crisis = s.inventory < 50.0
        s.reputation_crisis = s.brand_reputation < 25.0

        crisis_count = sum([s.cash_crisis, s.morale_crisis, s.inventory_crisis, s.reputation_crisis])
        if crisis_count > 0:
            logger.info("Q%d CRISES: cash=%s morale=%s inventory=%s reputation=%s",
                        s.quarter, s.cash_crisis, s.morale_crisis,
                        s.inventory_crisis, s.reputation_crisis)

        # ── Stage 13: Crisis Response ─────────────────────────────────────────
        # Only matters when crises are active. Two options: invest ot cut.
        if crisis_count > 0 and crisis_resp > 0:
            # Invest through crisis: spend cash to fix the underlying problem
            s.cash -= 500.0
            s.employee_morale += 5.0
            s.brand_reputation += 3.0
            self.action_labels.append("💰 Invested through crisis")
        elif crisis_count > 0 and crisis_resp < -0.3:
            # Emergency cost-cutting: save cash at expense of morale
            s.cash += 300.0
            s.employee_morale -= 5.0
            self.action_labels.append("✂️ Emergency cost-cutting")

        # ── Stage 13b: Intermediate Progress Signals ──────────────────────────
        # These signals track partial completion within large tasks (Stages 2 of 1)
        
        # 1. Metrics identified (Annual Report task)
        s.metrics_discovered = 0
        if s.revenue > 1000: s.metrics_discovered += 1
        if s.profit > 0: s.metrics_discovered += 1
        if s.customer_satisfaction > 70: s.metrics_discovered += 1
        if s.employee_morale > 70: s.metrics_discovered += 1
        if s.inventory > 100: s.metrics_discovered += 1
        if s.brand_reputation > 50: s.metrics_discovered += 1
        if s.rd_progress > 20: s.metrics_discovered += 1
        if s.total_employees > 15: s.metrics_discovered += 1
        
        # 2. Departments funded (Budget task)
        # Check if core departments have at least 10% budget share
        funded_depts = [
            s.finance_budget > 0.1,
            s.sales_budget > 0.1,
            s.hr_budget > 0.1,
            s.customer_budget > 0.1,
            s.ops_budget > 0.1
        ]
        departments_funded_count = sum(funded_depts)
        
        # 3. Negotiation Steps (Merger task)
        # Progress increases based on specific strategic actions
        if rd_invest > 100: s.merger_milestones = min(5, s.merger_milestones + 1)
        if task_alloc < -0.5: s.merger_milestones = min(5, s.merger_milestones + 1)
        if budget_shift > 0.5: s.merger_milestones = min(5, s.merger_milestones + 1)


        # ── Stage 14: Reward Calculation ──────────────────────────────────────
        # Reward = (how well things are going) - (how badly things are going)
        # Plus continuous shaping signals that reward improving trends.

        # Base positive components
        self.pos_reward = (
            max(0.0, s.profit / REWARD_NORM_PROFIT)
            + s.customer_satisfaction / REWARD_NORM_SATISFACTION
            + s.employee_morale / REWARD_NORM_MORALE
            + s.brand_reputation / REWARD_NORM_REPUTATION
            + (s.rd_progress / 100.0)  # R&D progress as direct bonus
        )

        # Base negative components (penalties for bad states)
        self.neg_reward = (
            max(0.0, -s.profit / REWARD_NORM_PROFIT)
            + max(0.0, (50.0 - s.employee_morale) / 15.0)
            + max(0.0, (30.0 - s.brand_reputation) / 20.0)
            + crisis_count * 0.5
        )

        # ── Continuous reward shaping (per-step improvement bonuses) ─────────
        # These small delta signals teach the AI to make progress each quarter,
        # not just optimise the final result.

        # 1. Profit improvement bonus: did we earn more than last quarter?
        profit_delta = (s.profit - self._prev_profit) / REWARD_NORM_PROFIT_DELTA
        profit_delta = float(np.clip(profit_delta, -1.0, 1.0))

        # 2. Morale improvement bonus: are employees happier than last quarter?
        morale_delta = (s.employee_morale - self._prev_morale) / REWARD_NORM_MORALE_DELTA
        morale_delta = float(np.clip(morale_delta, -1.0, 1.0))

        # 3. R&D payoff: ongoing bonus for high R&D progress (accumulated investment)
        rd_payoff = (s.rd_progress / 100.0) * REWARD_RD_SCALE

        # 4. Destructive action penalty: firing many people at once → negative signal
        fire_penalty = -REWARD_FIRE_PENALTY * (self._fired_this_step / 5.0)

        total_reward = float(
            self.pos_reward - self.neg_reward
            + profit_delta + morale_delta + rd_payoff + fire_penalty
        )

        logger.info(
            "Q%d Reward | total=%.3f pos=%.3f neg=%.3f Δprofit=%.3f Δmorale=%.3f "
            "rd_payoff=%.3f fire_penalty=%.3f",
            s.quarter, total_reward, self.pos_reward, self.neg_reward,
            profit_delta, morale_delta, rd_payoff, fire_penalty
        )

        # Store for next step delta calculations
        self._prev_profit = s.profit
        self._prev_morale = s.employee_morale

        # ── Stage 15: Termination Check ───────────────────────────────────────
        # Episode ends if: bankrupt, all customers gone, or less than 3 staff.
        terminated = (s.cash <= 0 or s.customer_satisfaction <= 5 or s.total_employees <= 2)
        truncated = s.quarter >= self.max_steps

        if terminated:
            # Massive terminal penalty — the AI must never "choose" to go bankrupt
            total_reward -= REWARD_TERMINAL_PENALTY
            logger.warning("Q%d EPISODE TERMINATED | cash=%.0f csat=%.0f employees=%d",
                           s.quarter, s.cash, s.customer_satisfaction, s.total_employees)

        # ── Post-step: Generate thought cloud, headline, record history ───────
        thought = self._generate_thought(s, crisis_count, action)
        if not self.action_labels:
            self.action_labels.append("Maintained current strategy")

        s.headline = self._generate_headline(s, crisis_count, total_reward)

        # Record structured per-step metrics (used by graders.py and CSV export)
        step_record: Dict[str, Any] = {
            "Quarter": s.quarter,
            "Cash": round(s.cash, 2),
            "Revenue": round(s.revenue, 2),
            "Profit": round(s.profit, 2),
            "Valuation": round(s.get_valuation(), 2),
            "Our_Price": round(our_price, 2),
            "Competitor_Price": round(s.competitor_price, 2),
            "Total Employees": s.total_employees,
            "Morale": round(s.employee_morale, 2),
            "Customer Satisfaction": round(s.customer_satisfaction, 2),
            "RD_Progress": round(s.rd_progress, 2),
            "Reward": round(total_reward, 4),
            "Metrics_Identified": s.metrics_discovered,
            "Departments_Funded": departments_funded_count,
            "Negotiation_Steps": s.merger_milestones,
            "Headline": s.headline,
            "AI Thought": thought.replace("\n", " "),
        }
        s.metrics_history.append(step_record)

        s.last_actions = {"actions": self.action_labels, "thought": thought}

        # Episode trace (for debugging — only written if TRACE_LOGGING=1)
        self.episode_trace.append(step_record)
        if (terminated or truncated) and os.getenv("TRACE_LOGGING") == "1":
            self._write_trace()

        info: Dict[str, Any] = {
            "pos_reward": self.pos_reward,
            "neg_reward": self.neg_reward,
            "reward_breakdown": {
                "profit_delta": profit_delta,
                "morale_delta": morale_delta,
                "rd_payoff": rd_payoff,
                "fire_penalty": fire_penalty,
            },
            "metrics_identified": s.metrics_discovered,
            "departments_funded_count": departments_funded_count,
            "negotiation_steps": s.merger_milestones,
            "thought": thought,
            "actions": self.action_labels,
            "crisis_count": crisis_count,
        }


        obs = s.to_observation()
        obs.reward = total_reward
        obs.done = terminated or truncated
        obs.info = info
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # state() — OpenEnv spec requires dict return
    # ──────────────────────────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        """
        Returns the current state as a plain dict (OpenEnv spec requirement).

        Includes all financial, operational, and HR metrics plus computed values
        like company valuation and department scores.
        """
        s = self.state_obj
        base = {
            "quarter": s.quarter,
            "cash": s.cash,
            "revenue": s.revenue,
            "profit": s.profit,
            "debt": s.debt,
            "customer_satisfaction": s.customer_satisfaction,
            "employee_morale": s.employee_morale,
            "inventory": s.inventory,
            "market_trend": s.market_trend,
            "competitor_price": s.competitor_price,
            "total_employees": s.total_employees,
            "avg_salary": s.avg_salary,
            "rd_progress": s.rd_progress,
            "brand_reputation": s.brand_reputation,
            "operational_efficiency": s.operational_efficiency,
            "cash_crisis": s.cash_crisis,
            "morale_crisis": s.morale_crisis,
            "inventory_crisis": s.inventory_crisis,
            "reputation_crisis": s.reputation_crisis,
            "news": s.news,
            "headline": s.headline,
            "valuation": s.get_valuation(),
            "dept_scores": s.dept_scores(),
        }
        return base

    def typed_state(self) -> State:
        """
        Returns the fully typed State object (for internal use and dashboard).
        Use state() for OpenEnv API compliance; use this for direct field access.
        """
        return self.state_obj

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _write_trace(self) -> None:
        """Write episode trace to JSON when TRACE_LOGGING=1 is set."""
        filename = f"episode_trace_q{self.state_obj.quarter}.json"
        try:
            with open(filename, "w") as f:
                json.dump(self.episode_trace, f, indent=2)
            logger.info("Episode trace written to %s", filename)
        except Exception as exc:
            logger.warning("Could not write trace: %s", exc)

    def _generate_thought(
        self, s: State, crisis_count: int, action: Optional[Action] = None
    ) -> str:
        """
        Generates the AI's causal reasoning in [Observation → Action → Result] format.
        This is shown in the 'Thought Cloud' on the dashboard.
        """
        # OBSERVATION — what's going on right now?
        if crisis_count >= 2:
            observation = f"Multiple Critical Metrics Failing ({crisis_count} crises)!"
        elif s.cash_crisis:
            observation = "Cash burn is too high, approaching bankruptcy."
        elif s.morale_crisis:
            observation = "Employees are burnt out, productivity risks are severe."
        elif s.inventory_crisis:
            observation = "Inventory stock empty, losing out on potential sales."
        elif s.reputation_crisis:
            observation = "Brand equity destroyed, customer demand plummeting."
        elif s.profit < 0:
            observation = "Operating at a net loss structurally."
        elif s.competitor_price < 40 and s.market_trend < 1.0:
            observation = "Competitors slashing prices in a shrinking market."
        elif s.profit > 1_000:
            observation = "Strong surplus margins generated."
        else:
            observation = "Normal business operations."

        # ACTION — what did we decide to do?
        chosen_action = "Maintaining standard allocations."
        if self.action_labels:
            non_hr = [lab for lab in self.action_labels if "Hired" not in lab and "Fired" not in lab]
            chosen_action = " | ".join(non_hr[:2]) if non_hr else self.action_labels[0]

        # RESULT — what happened as a consequence?
        if s.profit < 0:
            result = "Financials suffering. Bleeding cash reserves."
        elif s.rd_progress > 60:
            result = "R&D driving immense operational efficiency scaling."
        elif s.profit > 500:
            result = "Cash runway expanded, solid growth curve maintained."
        else:
            result = "Slight stabilisation achieved."

        return (
            f"**1. Observation**: {observation}\n\n"
            f"**2. Action**: {chosen_action}\n\n"
            f"**3. Result**: {result}"
        )

    def _generate_headline(
        self, s: State, crisis_count: int, reward: float
    ) -> str:
        """
        Generates a newspaper-style headline based on the current company situation.
        Shown at the top of the dashboard like a live news ticker.
        """
        # If a major world event just happened, lead with that
        if (s.news != "Company founded. Market stable."
                and s.event_history
                and f"Q{s.quarter}:" in s.event_history[-1]):
            return f"📰 {s.news}"

        if s.cash <= 0:
            return "📉 WALL STREET PANICS: Tech Giant Files for Bankruptcy!"
        elif crisis_count >= 2:
            return f"🚨 ACTIVIST INVESTORS CIRCLE: {crisis_count} Major Crises Paralyse CEO!"
        elif s.profit < -5_000:
            return "🔥 BLEEDING CASH: Shareholder Confidence Plummets Amidst Massive Losses."
        elif s.profit > 10_000 and s.employee_morale > 80:
            return "🏆 GOLDEN AGE: Unprecedented Profits Met With Soaring Employee Morale!"
        elif s.profit > 5_000:
            return "📈 SOLID EARNINGS: Cash Reserves Swell in Latest Financial Disclosures."
        elif reward < -2.0:
            return "⚠️ TROUBLE BREWING: Internal Metrics Flash Red as Growth Stalls."
        elif s.rd_progress > 80:
            return "🔬 INNOVATION HUB: R&D Division Unveils Next-Gen Technologies!"
        elif s.customer_satisfaction > 90:
            return "⭐ BELOVED BRAND: 'Can Do No Wrong', Says Satisfied Consumer Base."


    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="Autonomous CEO Simulator",
            description="Highly realistic, multi-stage corporate executive simulation.",
            version="2.1.0"
        )

