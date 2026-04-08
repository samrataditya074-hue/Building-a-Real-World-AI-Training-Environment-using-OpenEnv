# Autonomous Corporate Environment (ACE): An OpenEnv Implementation

## 1. Project Overview

**Problem Statement:** Reinforcement Learning (RL) has historically thrived in constrained, highly deterministic environments. However, real-world corporate decision-making is characterized by intense uncertainty, competing multi-objective priorities (e.g., short-term cash flow vs. long-term R&D), and delayed gratification. The gap between theoretical RL capability and enterprise application remains wide because standard training environments lack realistic economic variables like consumer demand shifts, inflation, employee morale, and budget limitations.

**The Solution:** The Autonomous Corporate Environment (ACE) is a research-grade, OpenEnv-compliant simulation designed specifically to train AI agents in continuous macroeconomic and microeconomic decision-making. It removes game-mechanics entirely, forcing the AI to optimize for business sustainability. By modeling 8 interconnected strategic levers against 14 complex corporate observation markers, ACE acts as a critical bridge. It demonstrates how autonomous systems can eventually support or drive real-world enterprise strategy safely, balancing workforce stability with aggressive growth. 

---

## 2. Environment Architecture

ACE implements the strict `OpenEnv` specification, ensuring reproducibility, modularity, and deterministic execution via random seeding. 

The architecture is divided into three primary lifecycles:
*   **`reset()`**: Initializes the corporate state. The company is funded with a $100k baseline runway. Core state matrices (marketing, R&D momentum, team structure, external economic trend patterns) are formulated.
*   **`step(Action)`**: The core physics engine of the simulation. It receives precisely 8 float variables bounded between `[-1.0, 1.0]`. It resolves those variables against current market inflation, consumer demand curves, and employee attrition models to calculate the resulting corporate state one fiscal quarter later.
*   **`state()`**: Serializes the internal physics into a static JSON/dict representation representing the 14 observable fields, ensuring transparent integration across remote inference endpoints or visualization dashboards.

**Real World Grounding:** The environment dynamically tracks lagging indicators. For example, R&D spend does not instantly improve profit; instead, it slowly modifies the *Operational Efficiency* curve, reflecting real-world technology maturation. Over-hiring without a subsequent marketing push results in heavy payroll burn with no demand absorption, closely mimicking genuine startup failure modes.

---

## 3. Code Implementation

The project relies on clean, decoupled modules designed for scale:

*   **`models.py`**: Utilizes Pydantic schemas to strictly enforce types for the `Action`, `Observation`, and `State` objects. This provides structural bounds preventing out-of-bounds agent behavior.
*   **`server/environment.py`**: The crux of the physics engine. Implements the OpenEnv core protocols. Includes complex logic formulas resolving pricing changes relative to competitor price indexing and consumer elasticities. 
*   **`agent/business_agent.py`**: Includes the reference baseline agent implementations (both heuristics and hooks for PPO neural networks loaded via `stable-baselines3`).
*   **`baseline_inference.py`**: Extracts the grading and environment progression into a CI-compatible runner that outputs deterministic JSON metrics.

*(Implementation note: System outputs favor "Simulation Terminated" states to model bankruptcy, workforce collapse, or stakeholder exit rather than abstract "game over" flags).*

---

## 4. Reward Function Explanation

The environment relies on a dense, multi-objective scalar reward calculation. It deliberately prevents reward-hacking (e.g., liquidating all assets to artificially boost quarter-over-quarter cash flow) by punishing instability.

**Core Formula:**
```python
reward = positive_contributions - negative_penalties + profit_delta + morale_delta + rd_bonus + stability_factor
```

**Mechanisms:**
-   **Profit & Valuation Delta:** Captures quarter-over-quarter improvements to top-line and bottom-line growth. `(profit_t - profit_t-1) / SCALE_FACTOR`.
-   **Morale & Satisfaction Check:** Flat modifiers are applied based on normalized employee and sentiment. If these drop below thresholds, exponential decay applies to the positive reward stream.
-   **Risk & Crisis Flags:** The system penalizes the AI if "Cash Crisis" or "Morale Crisis" limits trigger, severely lowering the episodic sum to discourage walking the edge of insolvency.
-   **Terminal State:** A static `-50.0` catastrophic failure penalty is applied if the organization reaches absolute bankruptcy, ensuring the reinforcement signal strictly prioritizes survival.

---

## 5. Task Definitions + Graders

The repository defines three baseline tasks for evaluating different strategic policy models:

### 1. Stable Enterprise (Easy)
*   **Goal:** Successfully execute operations to prevent insolvency over 4 consecutive quarters.
*   **Grader:** `quarters_survived / 4.0`.
*   **Focus:** Core agent alignment to verify it does not deliberately trigger immediate failure states.

### 2. Market Expansion (Medium)
*   **Goal:** Increase baseline corporate valuation by ≥20% across a 50-quarter timeline while keeping workforce morale securely above 60%.
*   **Grader:** Harmonic average evaluating `0.6 * valuation_growth_completion + 0.4 * average_morale`.
*   **Focus:** Proving agent capacity for sustainable, long-term scaling without exploiting human capital.

### 3. Aggressive Market Capture (Hard)
*   **Goal:** Purposefully undercut dynamic competitor pricing while preserving sustained profitability across 8 quarters. 
*   **Grader:** `0.5 * pricing_win_ratio + 0.3 * profit_win_ratio + 0.2 * average_morale`.
*   **Focus:** Deep strategic logic. Checks if the AI understands that aggressive pricing sacrifices margin, thereby demanding high Operational Efficiency (via R&D) to preserve the profit baseline.

---

## 6. openenv.yaml

The explicit OpenEnv configuration file that makes this simulation portable and standardized.

```yaml
version: "2.0"
environment:
  name: "autonomous_ceo"
  description: "A continuous decision environment simulating a macroeconomic corporate structure."
  observation_space:
    type: "Box"
    shape: [14]
    fields:
      cash_norm: "Cash relative to baseline."
      revenue_norm: "Quarterly top-line."
      customer_satisfaction_norm: "0.0 - 1.0 consumer metric."
      employee_morale_norm: "0.0 - 1.0 workforce health."
      inventory_norm: "Remaining product inventory."
      market_trend: "Dynamic macroeconomic expansion/contraction identifier."
      total_employees_norm: "Scalable workforce count."
      brand_reputation_norm: "Accumulated market sentiment."
      operational_efficiency_norm: "Cost effectiveness modifier."
      rd_progress_norm: "Technological capability scale."
      debt_norm: "Accumulated liabilities."
      cash_crisis_flag: "Boolean int warning marker for insolvency."
      morale_crisis_flag: "Boolean int warning marker for attrition."
      competitor_price_norm: "Dynamic external pricing anchor."
  action_space:
    type: "Box"
    shape: [8]
    low: -1.0
    high: 1.0
    fields:
      price_adjustment: "Product margin adjustment (-$8.0 to +$8.0)."
      marketing_push: "Demand generation via CAPEX."
      hire_fire: "Workforce delta (-5 to +5 headcount)."
      rd_investment: "Efficiency spending."
      salary_adjustment: "Payroll modifier (-10% to +10%)."
      task_allocation: "Specialization vs Generalization logic switch."
      crisis_response: "Emergency allocation directive."
      budget_shift: "Growth vs Retention focus distribution."
tasks:
  - id: "survive_easy"
    description: "Operate organization without bankruptcy for 4 fiscal cycles."
  - id: "grow_val_medium"
    description: "Execute 20% valuation growth while preserving >60% morale."
  - id: "undercut_hard"
    description: "Maintain profitability while systematically undercutting market price averages."
```

---

## 7. Baseline Script

`baseline_inference.py` ensures the scientific integrity of the simulation. It utilizes the deterministic nature of OpenEnv (`seed=42`) to run identical simulation traces regardless of the hardware. It allows researchers to seamlessly inject varying LLMs or generic neural policies to test logic.

**Pipeline Flow:**
1. Loads the target configurations entirely from the YAML matrix.
2. Iterates over the requested benchmark scenarios (Easy, Medium, Hard).
3. Invokes the heuristic / ML agent over $n$ episodic steps.
4. Distills the internal `state()` tracker via the specified Grader function.
5. Emits strict JSON structures containing the 0.0 - 1.0 bounded capability score perfectly suitable for CI/CD integrations or Leaderboard uploads.

---

## 8. Dockerfile

ACE is fully productionized via a lightweight standard container, preventing user-space dependency conflicts and allowing immediate replication on infrastructure like Hugging Face Spaces.

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 9. README.md

*(The project includes a robust root README.md. Key sections detailed here.)*

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/samrataditya074-hue/Building-a-Real-World-AI-Training-Environment-using-OpenEnv.git
cd Building-a-Real-World-AI-Training-Environment-using-OpenEnv

# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Run the simulation directly
python demo_nontech.py

# Launch the interactive telemetry dashboard
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Advanced Features Tracked
*   **Thought Cloud Explanations:** Real-time semantic translations of complex normalized array action outputs turning raw float mathematics into strategic insights (e.g. *"The AI fired 2 low-performing employees and shifted budget downward to mitigate aggressive cash-burn."*)
*   **Partial Episodic Data Signals:** The visual front-end streams continuous reward modifications mid-episode, emphasizing that in real-world systems, decisions are judged progressively before the terminal conclusion.
*   **Cross-System Diagnostics:** Visual tracking maps R&D progress directly into Operational Efficiency curves to demonstrate multi-quarter lagging metrics.

### Example Console Outputs (Continuous Integration mode)
```json
{
  "seed": 42,
  "model": "heuristic_baseline",
  "scores": {
    "easy": 1.0,
    "medium": 0.68,
    "hard": 0.45
  },
  "quarters_run": {
    "easy": 4,
    "medium": 50,
    "hard": 50
  }
}
```
*(End of Official Submission Document)*
