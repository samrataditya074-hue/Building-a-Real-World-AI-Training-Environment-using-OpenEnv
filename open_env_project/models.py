from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class Employee:
    """Individual employee in the company."""
    id: int
    name: str
    department: str  # Finance, Sales, HR, Customer, Ops
    performance: float  # 0-100
    salary: float
    morale: float  # 0-100
    tenure: int  # quarters worked

@dataclass
class Action:
    """
    Action space for the CEO AI. Each value should be carefully bounded between -1.0 and 1.0.
    """
    price_adjustment: float = 0.0          # -1 is max decrease, +1 is max increase
    marketing_push: float = 0.0            # 0 is no marketing, +1 is max marketing
    hire_fire: float = 0.0                 # -1 is max layoffs, +1 is max hiring
    rd_investment: float = 0.0             # 0 is no R&D, +1 is max R&D budget
    salary_adjustment: float = 0.0         # -1 is cut salaries, +1 is raise salaries
    task_allocation: float = 0.0           # -1 is specialize, +1 is generalize
    crisis_response: float = 0.0           # -1 is emergency cut, +1 is invest heavily
    budget_shift: float = 0.0              # -1 is conservative, +1 is aggressive growth

@dataclass
class Observation:
    """
    The state representation passed to the agent every step.
    Contains normalized continuous vectors or simple discrete flags to ease RL training.
    """
    cash_norm: float
    revenue_norm: float
    customer_satisfaction_norm: float
    employee_morale_norm: float
    inventory_norm: float
    market_trend: float
    total_employees_norm: float
    brand_reputation_norm: float
    operational_efficiency_norm: float
    rd_progress_norm: float
    debt_norm: float
    cash_crisis_flag: float
    morale_crisis_flag: float
    competitor_price_norm: float

    # OpenEnv standard fields since step() returns ObsT
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.cash_norm, self.revenue_norm, self.customer_satisfaction_norm,
            self.employee_morale_norm, self.inventory_norm, self.market_trend,
            self.total_employees_norm, self.brand_reputation_norm,
            self.operational_efficiency_norm, self.rd_progress_norm,
            self.debt_norm, self.cash_crisis_flag, self.morale_crisis_flag,
            self.competitor_price_norm
        ], dtype=np.float32)

@dataclass
class State:
    """
    The entire underlying truth/logic state of the business ecosystem.
    """
    cash: float = 200000.0
    revenue: float = 0.0
    profit: float = 0.0
    debt: float = 0.0

    customer_satisfaction: float = 75.0
    employee_morale: float = 80.0
    inventory: float = 500.0
    market_trend: float = 1.0
    competitor_price: float = 50.0

    total_employees: int = 20
    avg_salary: float = 3000.0
    rd_progress: float = 0.0  
    brand_reputation: float = 60.0  
    operational_efficiency: float = 70.0  

    finance_budget: float = 0.2
    sales_budget: float = 0.25
    hr_budget: float = 0.15
    customer_budget: float = 0.15
    ops_budget: float = 0.25

    cash_crisis: bool = False
    morale_crisis: bool = False
    inventory_crisis: bool = False
    reputation_crisis: bool = False

    news: str = "Company founded. Market stable."
    quarter: int = 0

    employees: List[Employee] = field(default_factory=list)
    last_actions: Dict[str, Any] = field(default_factory=dict)
    event_history: List[str] = field(default_factory=list)
    competitor_price_history: List[float] = field(default_factory=list)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    headline: str = "Welcome to your first Quarter as CEO!"

    def dept_scores(self) -> Dict[str, float]:
        """Department performance scores for visualization."""
        return {
            'Finance': min(10, self.cash / 2000),
            'Sales': min(10, self.revenue / 1000),
            'Customer': self.customer_satisfaction / 10,
            'HR': self.employee_morale / 10,
            'Operations': min(10, self.operational_efficiency / 10)
        }

    def get_valuation(self) -> float:
        """Calculates a holistic 'Company Valuation' metric for the growth trend chart."""
        return self.cash + (self.brand_reputation * 100) + (self.employee_morale * 10) + (self.rd_progress * 50)
        
    def get_roster(self) -> List[List[Any]]:
        """Returns employee roster as a list of lists for Gradio Dataframe."""
        roster = []
        for e in sorted(self.employees, key=lambda x: x.performance, reverse=True):
            roster.append([
                f"EMP-{e.id:03d}",
                e.name,
                e.department,
                f"${e.salary:.0f}",
                f"{e.performance:.1f}",
                f"{e.morale:.1f}",
                e.tenure
            ])
        return roster

    def to_observation(self) -> Observation:
        """Converts truth state into normalized agent representation."""
        return Observation(
            cash_norm=self.cash / 200000.0,
            revenue_norm=self.revenue / 5000.0,
            customer_satisfaction_norm=self.customer_satisfaction / 100.0,
            employee_morale_norm=self.employee_morale / 100.0,
            inventory_norm=self.inventory / 1000.0,
            market_trend=self.market_trend,
            total_employees_norm=self.total_employees / 50.0,
            brand_reputation_norm=self.brand_reputation / 100.0,
            operational_efficiency_norm=self.operational_efficiency / 100.0,
            rd_progress_norm=self.rd_progress / 100.0,
            debt_norm=self.debt / 10000.0,
            cash_crisis_flag=1.0 if self.cash_crisis else 0.0,
            morale_crisis_flag=1.0 if self.morale_crisis else 0.0,
            competitor_price_norm=self.competitor_price / 100.0,
        )
