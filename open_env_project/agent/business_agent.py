import numpy as np
import os

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

class CorporateAgent:
    """
    Autonomous CEO AI Agent.
    Makes multi-dimensional strategic decisions based on company state.
    Actions: [Price, Marketing, HireFire, RD, SalaryAdj, TaskAlloc, CrisisResponse, BudgetShift]
    """
    def __init__(self):
        self.prev_cash = 1.0
        self.prev_profit = 0.0
        self.growth_streak = 0
        self.decline_streak = 0
        self.model = None
        
        # Automatically load the trained RL agent if it exists
        if HAS_SB3 and os.path.exists("ceo_ppo_model.zip"):
            print("Loading Trained RL Model: ceo_ppo_model.zip")
            self.model = PPO.load("ceo_ppo_model.zip")

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        # If we have a smart RL model, use it!
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            
            # --- HACKATHON PRODUCTION GUARDRAILS ---
            # In real-world RL deployment, we use action masks to prevent catastrophic edge-cases
            # that haven't been fully trained out yet due to low timestep horizons.
            employees_norm = obs[6] # Normalized out of 50
            if employees_norm < 0.25: # If we have fewer than 12 employees, STRICTLY FORBID firing.
                action[2] = max(0.2, action[2]) # Force stable hiring
                
            cash_norm = obs[0]
            if cash_norm < 0.1: # Less than 20k cash
                action[7] = min(-0.5, action[7]) # Force conservative budget shift
            
            return action

        # Fallback: Naive Heuristic Hand-coded Strategy
        cash      = obs[0]   # normalized
        revenue   = obs[1]
        csat      = obs[2]
        morale    = obs[3]
        inventory = obs[4]
        trend     = obs[5]
        employees = obs[6]
        reputation= obs[7]
        efficiency= obs[8]
        rd_prog   = obs[9]
        debt      = obs[10]
        cash_cris = obs[11]
        morale_cr = obs[12]
        competitor_price = obs[13] if len(obs) > 13 else 0.5

        # Track momentum
        cash_delta = cash - self.prev_cash
        if cash_delta > 0:
            self.growth_streak += 1
            self.decline_streak = 0
        else:
            self.decline_streak += 1
            self.growth_streak = 0
        self.prev_cash = cash

        # === CRISIS MODE ===
        if cash_cris > 0.5:
            return self._crisis_cash(cash, morale, inventory, employees)
        if morale_cr > 0.5:
            return self._crisis_morale(cash, morale, employees)

        # === GROWTH MODE ===
        if self.growth_streak > 3 and cash > 0.8:
            return self._growth_strategy(cash, csat, morale, inventory, trend, rd_prog, reputation)

        # === DECLINE RECOVERY ===
        if self.decline_streak > 3:
            return self._recovery_strategy(cash, morale, inventory, efficiency)

        # === STANDARD OPERATIONS ===
        return self._standard_strategy(cash, revenue, csat, morale, inventory, trend, rd_prog, efficiency, reputation)

    def _crisis_cash(self, cash, morale, inventory, employees):
        """Emergency: preserve cash at all costs."""
        price = 0.6                    # raise prices aggressively
        marketing = -0.3               # cut marketing
        hire_fire = -0.8 if employees > 0.3 else 0.0  # layoffs
        rd = -0.5                      # cut R&D
        salary = -0.08                 # salary cuts
        task = -0.5                    # specialize to maximize output
        crisis = -0.8                  # aggressive cost cutting
        budget = -0.7                  # conservative
        return np.array([price, marketing, hire_fire, rd, salary, task, crisis, budget], dtype=np.float32)

    def _crisis_morale(self, cash, morale, employees):
        """Emergency: boost morale before mass exodus."""
        price = 0.0
        marketing = 0.1
        hire_fire = 0.4 if cash > 0.5 else 0.0  # hire to reduce workload
        rd = 0.1
        salary = 0.08                  # raise salaries
        task = 0.6                     # cross-train for variety
        crisis = 0.7                   # invest through crisis
        budget = 0.0
        return np.array([price, marketing, hire_fire, rd, salary, task, crisis, budget], dtype=np.float32)

    def _growth_strategy(self, cash, csat, morale, inventory, trend, rd_prog, reputation):
        """Capitalize on momentum with expansion."""
        price = -0.2 if csat > 0.7 else 0.1
        marketing = 0.7               # heavy marketing push
        hire_fire = 0.6               # aggressive hiring
        rd = 0.8 if rd_prog < 0.6 else 0.3  # invest in innovation
        salary = 0.05                 # modest raises
        task = 0.4                    # flexible teams
        crisis = 0.0
        budget = 0.6                  # aggressive budget
        return np.array([price, marketing, hire_fire, rd, salary, task, crisis, budget], dtype=np.float32)

    def _recovery_strategy(self, cash, morale, inventory, efficiency):
        """Stabilize after decline."""
        price = 0.3                   # raise prices slightly
        marketing = 0.2              # moderate marketing
        hire_fire = -0.3 if cash < 0.4 else 0.0
        rd = 0.1
        salary = 0.0                 # freeze salaries
        task = -0.6                  # specialize for efficiency
        crisis = 0.3 if cash > 0.5 else -0.5
        budget = -0.3               # conservative
        return np.array([price, marketing, hire_fire, rd, salary, task, crisis, budget], dtype=np.float32)

    def _standard_strategy(self, cash, revenue, csat, morale, inventory, trend, rd_prog, efficiency, reputation):
        """Balanced day-to-day operations."""
        # Price: lower if inventory high, raise if cash low
        price = -0.3 if inventory > 0.8 else (0.3 if cash < 0.5 else 0.05)

        # Marketing: scale with available cash
        marketing = 0.5 if cash > 0.6 else 0.15

        # Hiring: hire if morale low and cash ok, or if understaffed
        hire_fire = 0.3 if (morale < 0.6 and cash > 0.6) else (-0.2 if cash < 0.3 else 0.0)

        # R&D: invest when stable
        rd = 0.4 if (csat > 0.6 and cash > 0.7 and rd_prog < 0.7) else 0.1

        # Salary: small raises if morale dipping
        salary = 0.03 if morale < 0.5 else (-0.02 if cash < 0.4 else 0.0)

        # Task allocation: specialize if efficiency low
        task = -0.4 if efficiency < 0.6 else 0.2

        # No crisis
        crisis = 0.0

        # Budget: moderate
        budget = 0.2 if trend > 1.1 else (-0.2 if trend < 0.8 else 0.0)

        return np.array([price, marketing, hire_fire, rd, salary, task, crisis, budget], dtype=np.float32)
