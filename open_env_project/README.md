# 🏢 Autonomous CEO AI Simulator - OpenEnv

The **Autonomous CEO AI Simulator** is a fully transparent, highly visual, and OpenEnv-compliant Reinforcement Learning environment designed for testing and visualizing sequential decision-making in a dynamic corporate simulation.

This package was built focusing strongly on **Explainability** and **Live Visualization** — allowing judges, developers, and RL theorists to watch standard Continuous Action Space algorithms explicitly narrate their strategy in real-time.

## 🚀 Hackathon Features

1. **Macro Action Space / Micro Narrative Engine**:
    The API consumes an 8-float Action space (stable and highly predictable for PPO/SAC). However, the internal Environment Engine translates these floats into targeted micro-actions (e.g. `hire_fire=-0.5` becomes **"Fired lowest performers: Taylor (HR), Jordan (Ops)"**), ensuring deep explainability without action-space combinatorial explosion.

2. **Causal Thought Cloud**:
    The AI narratively justifies every step through an explicitly structured 3-phase reasoning frame: `Observation -> Action -> Result`.

3. **Ultimate Executive Dashboard**:
    Watch your agent train via an immersive single-page command center running on Gradio + Plotly:
    - **Corporate Valuation Trend Plot**: Watch overarching company health growth vs downfall.
    - **Live Employee Roster**: A streaming `Pandas`-style table detailing individual employee names, roles, and granular performance metrics modified by the AI in real-time.
    - **Pricing War Plot**: Track the dynamic competitor AI aggressively reacting to your pricing adjustments over 100 quarters.
    - **Animated 3D Ecosystem Map**: Departmental performance surfaces shifting as funding alters.

## 🏗 Schema Design

### State & Observation
The raw truth is housed in the `State` dataclass (which includes explicit details for all 20+ unique `Employee` entities, detailed global crisis thresholds, and historical event logs).

This compresses down into the `Observation` Space natively returned to Agents: 14 bounds-normalized (`[0..2]`) floating/boolean values representing Cash, Revenue, Customer Satisfaction, R&D Progress, and Global Crises.

### Action Levers (8-dim Continuous Box `[-1.0, 1.0]`)
1. **Pricing**: Increase/Decrease unit pricing.
2. **Marketing Push**: Ad spend allocation volume.
3. **Hire/Fire Volume**: Dictates scaling or downsizing (engine auto-targets high/low performers).
4. **R&D Innovation**: Investment allocation for long-term operational efficiency.
5. **Salary Adjustment**: Implements performance-based bonuses (+) or company-wide cuts (-).
6. **Task Allocation**: Generalize teams (+) vs Specialize teams (-).
7. **Crisis Response**: Capital injection (+) vs Emergency downsizing (-).
8. **Budget Shift**: Aggressive growth shifts (+) vs Conservatism (-).

### Reward Function
$$ R_{total} = R_{positive} - R_{negative} $$
The environment natively streams individual `pos_reward` and `neg_reward` integers through the `StepResult` info dict.
- **Growth (+)**: Profitability, Morale, R&D Progress, Customer Success.
- **Risk (-)**: Net Losses, High Employee Turnover, Inventory depletion.

### The 3 Tasks (Grading Modules)
Using OpenEnv Tasks methodology (`tasks.py`), we enforce standardized capability benchmarks:
1. **Easy (`grade_survival`)**: Can the policy just survive 100 quarters without total collapse?
2. **Medium (`grade_profit`)**: Objective-based scoring; hits max return at $10k positive cashflow.
3. **Hard (`grade_utopia`)**: Rewards perfection; heavily penalized if Morale or R&D drops during the scaling path.

## 📦 Running & Deployment

### Run Visually (Gradio Hackathon Demo)
Start the `uvicorn` web server that binds both the OpenEnv protocol endpoints and the frontend:
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Then navigate to `http://localhost:7860/` for the Single Page CEO Dashboard.

### Train the AI via Gymnasium Wrapper
Launch a 100k step PPO agent loop via Stable-Baselines3 natively natively targeting the RL Engine:
```bash
pip install stable-baselines3 shimmy gymnasium
python train_rl.py
```

### Deploy to Hugging Face Spaces
We mapped OpenEnv standard structure. Deploy the containerized architecture seamlessly via standard cli:
```bash
# See deploy.ps1
openenv push --space-name autonomous-ceo
```
