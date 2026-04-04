import random
import numpy as np
from typing import Dict, Any, Tuple
from openenv.core.env_server import Environment

from models import Action, Observation, State, Employee

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
               "Blake", "Drew", "Sage", "Reese", "Dakota", "Finley", "Rowan", "Skyler",
               "Charlie", "Emerson", "Hayden", "Parker", "Jamie", "Peyton", "Kai", "Tatum"]
DEPARTMENTS = ["Finance", "Sales", "HR", "Customer", "Ops"]

def _make_employee(eid: int) -> Employee:
    return Employee(
        id=eid,
        name=random.choice(FIRST_NAMES),
        department=random.choice(DEPARTMENTS),
        performance=random.uniform(40, 95),
        salary=random.uniform(2000, 5000),
        morale=random.uniform(50, 95),
        tenure=0,
    )

class CEOEnvironment(Environment[Action, Observation, State]):
    """
    OpenEnv compliant Autonomous CEO AI Simulator Environment.
    """
    def __init__(self):
        super().__init__()
        self.max_steps = 200
        self.state_obj: State = State()
        self.pos_reward = 0.0
        self.neg_reward = 0.0
        self.action_labels = []

    def reset(self, **kwargs) -> Observation:
        employees = [_make_employee(i) for i in range(20)]
        self.state_obj = State(employees=employees)
        self.state_obj.competitor_price_history.append(self.state_obj.competitor_price)
        self.action_labels = []
        obs = self.state_obj.to_observation()
        obs.reward = 0.0
        obs.done = False
        obs.info = {}
        return obs

    def step(self, action: Action) -> Observation:
        s = self.state_obj
        s.quarter += 1
        self.action_labels = []

        # --- Decode 8 Actions ---
        price_adj    = action.price_adjustment * 8.0
        mkt_spend    = max(0, action.marketing_push * 600.0)
        hire_fire    = int(round(action.hire_fire * 5))
        rd_invest    = max(0, action.rd_investment * 400.0)
        salary_adj   = action.salary_adjustment * 0.10      # ±10% salary change
        task_alloc   = action.task_allocation              # -1=specialize, +1=generalize
        crisis_resp  = action.crisis_response              # -1=cut, +1=invest through crisis
        budget_shift = action.budget_shift                 # -1=conservative, +1=aggressive

        # --- 1. Market Dynamics & Dynamic Events ---
        s.market_trend += np.random.normal(0, 0.06)
        s.market_trend = float(np.clip(s.market_trend, 0.4, 1.6))
        
        event_msg = None
        shock = random.random()
        if shock < 0.015:
            s.market_trend *= 0.5
            s.customer_satisfaction -= 15
            event_msg = "📉 GLOBAL RECESSION! Markets plunge, consumer spending halted."
        elif shock > 0.985:
            s.market_trend *= 1.4
            s.brand_reputation = min(100, s.brand_reputation + 20)
            event_msg = "🚀 ECONOMIC BOOM! Golden quarter for the industry!"
        elif 0.015 <= shock <= 0.03:
            s.inventory = max(0, s.inventory - 500)
            s.operational_efficiency = max(20, s.operational_efficiency - 20)
            event_msg = "⛓️ LABOR STRIKE! Operations halted, massive supply chain shock."
        elif 0.97 <= shock <= 0.985:
            s.rd_progress += 25
            s.operational_efficiency += 15
            event_msg = "🧬 TECH BREAKTHROUGH! R&D discovers revolutionary workflow."
        elif 0.03 <= shock <= 0.045:
            s.competitor_price *= 1.3
            s.market_trend *= 1.1
            event_msg = "🏛️ COMPETITOR SCANDAL! Rival company investigated, customers flock to us!"
            
        if event_msg:
            s.news = event_msg
            s.event_history.append(f"Q{s.quarter}: {event_msg}")

        # --- 2. HR: Hire / Fire ---
        if hire_fire > 0:
            hired_names = []
            for _ in range(hire_fire):
                new_emp = _make_employee(len(s.employees))
                s.employees.append(new_emp)
                hired_names.append(new_emp.name)
            self.action_labels.append(f"Hired {hire_fire}: {', '.join(hired_names)}")
        elif hire_fire < 0:
            fires = min(abs(hire_fire), len(s.employees))
            if fires > 0:
                worst = sorted(s.employees, key=lambda e: e.performance)[:fires]
                fired_names = []
                for w in worst:
                    s.employees.remove(w)
                    fired_names.append(f"{w.name} ({w.department})")
                self.action_labels.append(f"Fired lowest performers: {', '.join(fired_names)}")
                s.employee_morale -= fires * 3.0

        s.total_employees = len(s.employees)

        # --- 3. Salary Adjustments ---
        if abs(salary_adj) > 0.01:
            bonused_names = []
            for emp in s.employees:
                if salary_adj > 0 and emp.performance > 75:  # Merit-based bonus distribution
                    emp.salary *= (1.0 + salary_adj * 2) 
                    emp.morale += 15
                    bonused_names.append(emp.name)
                elif salary_adj < 0:
                    emp.salary *= (1.0 + salary_adj)
                    emp.morale -= 8
                emp.salary = max(1500, min(emp.salary, 10000))

            if salary_adj > 0 and bonused_names:
                self.action_labels.append(f"Performance Bonuses granted to top {len(bonused_names)}: {', '.join(bonused_names[:3])}...")
            elif salary_adj < 0:
                self.action_labels.append(f"Company-wide Salary Cut: {abs(salary_adj)*100:.0f}%")

        # --- 4. Task / Team Allocation ---
        if task_alloc > 0.3:
            s.operational_efficiency += 2.0
            self.action_labels.append("Cross-trained teams for flexibility")
        elif task_alloc < -0.3:
            s.operational_efficiency += 3.5
            for emp in s.employees:
                emp.performance += 1.5
            self.action_labels.append("Specialized departments for focus")

        # --- 5. Employee Dynamics (performance, morale, tenure) ---
        for emp in s.employees:
            emp.tenure += 1
            emp.performance += random.uniform(-2, 2) + (emp.morale - 50) * 0.02
            emp.performance = float(np.clip(emp.performance, 10, 100))
            emp.morale += random.uniform(-3, 3) + salary_adj * 10
            emp.morale = float(np.clip(emp.morale, 5, 100))

        if s.employees:
            s.employee_morale = float(np.mean([e.morale for e in s.employees]))
            s.avg_salary = float(np.mean([e.salary for e in s.employees]))

        # --- 6. Budget Allocation ---
        shift = budget_shift * 0.03
        s.sales_budget = float(np.clip(s.sales_budget + shift, 0.05, 0.5))
        s.ops_budget = float(np.clip(s.ops_budget - shift * 0.5, 0.05, 0.5))
        if abs(budget_shift) > 0.3:
            mode = "aggressive growth" if budget_shift > 0 else "conservative savings"
            self.action_labels.append(f"Budget shifted to {mode}")

        # --- 7. Customer Demand ---
        # Reactive Competitor Logic
        our_price = s.competitor_price + price_adj  # Using competitor_price as the base
        s.competitor_price = s.competitor_price * 0.8 + our_price * 0.2 + np.random.normal(0, 1.0)
        s.competitor_price = float(np.clip(s.competitor_price, 20.0, 150.0))
        s.competitor_price_history.append(s.competitor_price)

        # Rebalanced Economy: Demand scales up high enough to support payroll costs
        base_demand = 450.0 * s.market_trend * (s.total_employees / 20.0)
        price_factor = max(0.2, 2.0 - (our_price / max(1, s.competitor_price)))
        mkt_bonus = mkt_spend / 10.0  # $10 of marketing roughly gets 1 extra customer
        rep_bonus = s.brand_reputation * 1.5
        demand = max(0, base_demand * price_factor + mkt_bonus + rep_bonus)

        # --- 8. Operations ---
        sales = min(s.inventory, demand)
        s.inventory -= sales
        
        # Production scales with workforce size and operational efficiency
        production = (s.total_employees * 25.0) + (s.operational_efficiency * 3.0) 
        s.inventory += production
        s.inventory = max(0, s.inventory)
        s.revenue = sales * our_price
        if mkt_spend > 200:
            self.action_labels.append(f"Marketing push: ${mkt_spend:.0f}")

        # --- 9. R&D Progress ---
        if rd_invest > 50:
            s.rd_progress += rd_invest / 100.0
            s.rd_progress = min(100, s.rd_progress)
            self.action_labels.append(f"R&D investment: ${rd_invest:.0f}")
        if s.rd_progress > 30:
            s.operational_efficiency += 0.5
        if s.rd_progress > 70:
            s.brand_reputation += 0.3

        # --- 10. Financials ---
        payroll = s.avg_salary * s.total_employees / 4.0  # quarterly
        costs = 400.0 + mkt_spend + rd_invest + payroll
        s.profit = s.revenue - costs
        s.cash += s.profit
        if price_adj != 0:
            self.action_labels.append(f"Price {'↑' if price_adj > 0 else '↓'} ${abs(price_adj):.1f}")

        # --- 11. Reputation & Satisfaction ---
        s.customer_satisfaction += (mkt_spend / 800.0) - (price_adj / 15.0) + (s.operational_efficiency - 70) * 0.02
        s.customer_satisfaction = float(np.clip(s.customer_satisfaction, 0, 100))
        s.brand_reputation += s.profit / 5000.0
        s.brand_reputation = float(np.clip(s.brand_reputation, 0, 100))
        s.operational_efficiency = float(np.clip(s.operational_efficiency, 20, 100))

        # --- 12. Crisis Detection ---
        s.cash_crisis = s.cash < 2000
        s.morale_crisis = s.employee_morale < 35
        s.inventory_crisis = s.inventory < 50
        s.reputation_crisis = s.brand_reputation < 25

        # --- 13. Crisis Response ---
        crisis_count = sum([s.cash_crisis, s.morale_crisis, s.inventory_crisis, s.reputation_crisis])
        if crisis_count > 0 and crisis_resp > 0:
            s.cash -= 500  # invest through crisis
            s.employee_morale += 5
            s.brand_reputation += 3
            self.action_labels.append("💰 Invested through crisis")
        elif crisis_count > 0 and crisis_resp < -0.3:
            s.cash += 300
            s.employee_morale -= 5
            self.action_labels.append("✂️ Emergency cost-cutting")

        # --- 14. Rewards ---
        self.pos_reward = (
            max(0, s.profit / 80.0) +
            s.customer_satisfaction / 40.0 +
            s.employee_morale / 60.0 +
            s.brand_reputation / 80.0 +
            (s.rd_progress / 100.0)
        )
        self.neg_reward = (
            max(0, -s.profit / 80.0) +
            max(0, (50 - s.employee_morale) / 15.0) +
            max(0, (30 - s.brand_reputation) / 20.0) +
            crisis_count * 0.5
        )
        total_reward = float(self.pos_reward - self.neg_reward)

        # --- 15. Termination & Penalty ---
        terminated = s.cash <= 0 or s.customer_satisfaction <= 5 or s.total_employees <= 2
        truncated = s.quarter >= self.max_steps

        # Prevent the AI from executing a "suicide strategy" (firing everyone to avoid payroll losses)
        if terminated:
            total_reward -= 50.0  # Massive terminal penalty for killing the company

        thought = self._generate_thought(s, crisis_count, action)
        if not self.action_labels:
            self.action_labels.append("Maintained current strategy")
            
        s.headline = self._generate_headline(s, crisis_count, total_reward)
        s.metrics_history.append({
            "Quarter": s.quarter,
            "Cash": round(s.cash, 2),
            "Revenue": round(s.revenue, 2),
            "Profit": round(s.profit, 2),
            "Valuation": round(s.get_valuation(), 2),
            "Total Employees": s.total_employees,
            "Morale": round(s.employee_morale, 2),
            "Customer Satisfaction": round(s.customer_satisfaction, 2),
            "Headline": s.headline,
            "AI Thought": thought.replace('\n', ' ')
        })

        s.last_actions = { "actions": self.action_labels, "thought": thought }

        info = {
            "pos_reward": self.pos_reward,
            "neg_reward": self.neg_reward,
            "thought": thought,
            "actions": self.action_labels,
            "crisis_count": crisis_count,
        }

        obs = s.to_observation()
        obs.reward = total_reward
        obs.done = terminated or truncated
        obs.info = info

        return obs

    def state(self) -> State:
        return self.state_obj

    def _generate_thought(self, s: State, crisis_count: int, action: Action = None) -> str:
        """
        Hackathon Requirement: Thought Cloud follows specific format:
        [Observation] -> [Action] -> [Result]
        """
        observation = "Normal business operations."
        chosen_action = "Maintaining standard allocations."
        result = "Awaiting market feedback."

        # OBSERVATION
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
        elif s.profit > 1000:
            observation = "Strong surplus margins generated."

        # ACTION (Infer intent from Action object if passed, else fallback to labels)
        if self.action_labels:
            # Join top 2 distinctive actions
            chosen_action = " | ".join([lab for lab in self.action_labels if "Hired" not in lab and "Fired" not in lab][:2])
            if not chosen_action: 
                chosen_action = self.action_labels[0]
        
        # RESULT (Look at state deltas or profit)
        if s.profit < 0:
            result = "Financials suffering. Bleeding cash reserves."
        elif s.rd_progress > 60:
            result = "R&D driving immense operational efficiency scaling."
        elif s.profit > 500:
            result = "Cash runway expanded, solid growth curve maintained."
        else:
            result = "Slight stabilization achieved."

        return f"**1. Observation**: {observation}\n\n**2. Action**: {chosen_action}\n\n**3. Result**: {result}"

    def _generate_headline(self, s: State, crisis_count: int, reward: float) -> str:
        """Generates dynamic newspaper headlines based on performance."""
        if s.news != "Company founded. Market stable." and f"Q{s.quarter}:" in (s.event_history[-1] if s.event_history else ""):
            return f"📰 {s.news}" # Override with major event shock
            
        if s.cash <= 0:
            return "📉 WALL STREET PANICS: Tech Giant Files for Bankruptcy!"
        elif crisis_count >= 2:
            return f"🚨 ACTIVIST INVESTORS CIRCLE: {crisis_count} Major Crises Paralyze CEO!"
        elif s.profit < -5000:
            return "🔥 BLEEDING CASH: Shareholder Confidence Plummets Amidst Massive Losses."
        elif s.profit > 10000 and s.employee_morale > 80:
            return "🏆 GOLDEN AGE: Unprecedented Profits Met With Soaring Employee Morale!"
        elif s.profit > 5000:
            return "📈 SOLID EARNINGS: Cash Reserves Swell in Latest Financial Disclosures."
        elif reward < -2.0:
            return "⚠️ TROUBLE BREWING: Internal Metrics Flash Red as Growth Stalls."
        elif s.rd_progress > 80:
            return "🔬 INNOVATION HUB: R&D Division Unveils Next-Gen Technologies!"
        elif s.customer_satisfaction > 90:
            return "⭐ BELOVED BRAND: 'Can Do No Wrong', Says Satisfied Consumer Base."
        
        return "📊 MARKET WATCH: Company Holds Steady in Routine Quarter."
