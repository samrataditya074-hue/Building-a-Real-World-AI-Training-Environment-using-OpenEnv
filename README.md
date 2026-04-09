---
title: Autonomous CEO AI Simulator
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏢 Autonomous CEO AI Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-v2.0-green.svg)](https://openenv.dev)

> **In plain English**: This is an AI that runs a company. Every three months (a "quarter") it makes 8 critical decisions (pricing, hiring, R&D, etc.) exactly like a real CEO would. You can watch it think and act live in your browser, or race against it yourself in Clash Mode.

---

## 🎯 Judges — Start Here

Welcome! If you only have 2 minutes, here is what you need to know:
1. We built a highly realistic, OpenEnv-compliant corporate simulator that mathematically models an entire economy, supply chain, and employee ecosystem.
2. We trained a Reinforcement Learning AI capable of surviving and optimizing this environment autonomously.
3. **The WOW Factor**: Our interactive 3D dashboard utilizes live LLM inferences to natively translate the agent's math into a **"Thought Cloud"**—meaning you can read the AI's strategic reasoning in plain English while it actively runs the company.

### 🚀 Instant Demo (Local)

1. Ensure you have python installed. Clone this repo and run:
```bash
pip install -r requirements.txt
python demo_nontech.py
```
2. To launch the full HD 3D Dashboard:
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# Go to http://localhost:7860
```


---

## 📋 What Is This?

The **Autonomous CEO AI Simulator** is an [OpenEnv](https://openenv.dev)-compliant Reinforcement Learning environment where an AI agent acts as a CEO, making **8 quarterly strategic decisions** to grow a virtual company.

It was built for the **Scaler Hackathon** with three goals:
1. **Demonstrate powerful AI decision-making** in a realistic, relatable scenario
2. **Explain every decision** in plain English (the "Thought Cloud")
3. **Let you compete** against the AI in real-time

### Key Features
- 🤖 **AI CEO** — makes 8 continuous strategic decisions every quarter
- 📊 **Live Executive Dashboard** — 3D charts, pricing war, reward analytics
- ⚔️ **CEO Clash Mode** — human vs AI, 50-quarter race
- 🏆 **Leaderboard** — tracks best company valuations across runs
- 📰 **Newspaper Headlines** — dynamic story of your company's journey
- 🔬 **Three Graded Tasks** — easy → medium → hard benchmarks

---

## 🎛️ Action / Observation Cheat Sheet

### 8 Action Levers (what the AI decides each quarter)

| # | Lever | -1.0 means | +1.0 means | Real impact |
|---|---|---|---|---|
| 1 | 💰 `price_adjustment` | Lower price by $8 | Raise price by $8 | Affects demand and margin |
| 2 | 📣 `marketing_push` | $0 marketing | $600 marketing spend | Boosts demand & satisfaction |
| 3 | 👥 `hire_fire` | Fire 5 worst performers | Hire 5 people | Changes team size & payroll |
| 4 | 🧬 `rd_investment` | No R&D spend | $400 R&D budget | Long-term efficiency gains |
| 5 | 💵 `salary_adjustment` | -10% pay cuts | +10% merit bonuses | Affects morale & cost |
| 6 | 📋 `task_allocation` | Specialise teams | Cross-train teams | Efficiency vs flexibility |
| 7 | 🚨 `crisis_response` | Emergency cost cuts | Invest through crisis | Only active during crises |
| 8 | 📊 `budget_shift` | Conservative savings | Aggressive growth | Reallocates between Sales/Ops |

*Note: The mathematical RL bounds outputs are uniformly `[-1, 1]`. The simulation automatically maps properties strictly required to be non-negative (like R&D and Marketing) linearly to `[0, 1]` using the `scale_positive` function.*

### 14 Observation Fields (what the AI sees each quarter)

| # | Field | Range | What it represents |
|---|---|---|---|
| 0 | `cash_norm` | 0–2 | Cash ÷ $200,000. 1.0 = starting level |
| 1 | `revenue_norm` | 0–2 | Quarterly revenue ÷ $5,000 |
| 2 | `customer_satisfaction_norm` | 0–1 | Customer happiness (0–100%) |
| 3 | `employee_morale_norm` | 0–1 | Average employee happiness (0–100%) |
| 4 | `inventory_norm` | 0–2 | Units in stock ÷ 1,000 |
| 5 | `market_trend` | 0.4–1.6 | Economy health (1.0 = neutral) |
| 6 | `total_employees_norm` | 0–1 | Employees ÷ 50 |
| 7 | `brand_reputation_norm` | 0–1 | Brand trust score (0–100%) |
| 8 | `operational_efficiency_norm` | 0–1 | How efficiently operations run |
| 9 | `rd_progress_norm` | 0–1 | Technology advancement (0–100%) |
| 10 | `debt_norm` | 0–2 | Debt ÷ $10,000 |
| 11 | `cash_crisis_flag` | 0 or 1 | 1 = cash below $2,000 |
| 12 | `morale_crisis_flag` | 0 or 1 | 1 = morale below 35% |
| 13 | `competitor_price_norm` | 0–1.5 | Competitor price ÷ $100 |

---

## 📦 How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Non-technical plain-English demo

```bash
python demo_nontech.py
# Optional flags:
python demo_nontech.py --quarters 10 --seed 99
```

---

## 🛠 Usage & Deployment

### 1. Mandatory Configuration
Before running inference or submitting, ensure the following environment variables are set:
- `API_BASE_URL`: The API endpoint for the LLM (default: `https://router.huggingface.co/v1`).
- `MODEL_NAME`: The model identifier to use (default: `Qwen/Qwen2.5-72B-Instruct`).
- `HF_TOKEN`: Your Hugging Face / API key.

### 2. Running Validation
To ensure the environment is fully compliant with OpenEnv Phase 2 specs, run:
```bash
openenv validate
```
**Expected Output:**
```text
[OK] Autonomous CEO Simulator: Ready for multi-mode deployment
```

### 3. Local task Evaluation (Baseline)
Run the automated evaluation script to check task completion and grader scoring:
```bash
# Set your HF_TOKEN first
export HF_TOKEN="your_token_here"
python inference.py
```
*Note: The script emits structured logs strictly matching the `[START]`, `[STEP]`, and `[END]` format required by hackathon judges.*

### 4. Docker Build & Deployment
Build the container for production or Hugging Face Spaces:
```bash
docker build -t ceo-sim .
```

### 5. Training the Agent
To train a new RL policy using Stable-Baselines3:
```bash
python train_rl.py
```


### Step 5 — Hugging Face Spaces Deployment (Free Cloud Hosting)

This project is perfectly crafted to deploy instantly to Hugging Face Spaces via Docker. No internal file changes are needed.

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and select **Docker** as your Space SDK.
2. Select the **Blank** Docker template.
3. Upload all files from this repository.
4. *(Optional)* Go to Settings -> Variables and secrets -> Add your `GROQ_API_KEY` for the live LLM thought feature.
5. The cloud will automatically build the `Dockerfile` and expose port `7860`. Your app will be live globally!

#### Local Docker Testing
```bash
docker build -t ceo-sim .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... ceo-sim
```

### Step 5 — Baseline evaluation (all 3 tasks)

```bash
python baseline_inference.py --seed 42
# With Groq (Live LLM Insights):
GROQ_API_KEY=gsk_... python baseline_inference.py --seed 42 --model llama-3.3-70b-versatile
```

### Step 6 — Train the AI yourself (optional, takes hours)

```bash
python train_rl.py
```

### Step 7 — Run the test suite

```bash
pytest tests/ -v
```

---

## 🏆 Tasks & Scoring

The environment defines three benchmark tasks, each returning a score in **[0.0 → 1.0]**:

| Task | Difficulty | Goal | How it's scored |
|---|---|---|---|
| **Annual Report** | 🟢 Easy | Identify key performance metrics in the annual report | `0.4 × survival + 0.6 × metrics_identified` |
| **Budget** | 🟡 Medium | Ensure all 5 departments are adequately funded (>10%) | `count(funded_depts) / 5` (average per step) |
| **Merger** | 🔴 Hard | Lead a multi-stage negotiation for a strategic merger | `negotiation_steps / 5` |
| **Market Strategy**| 🔴 Hard | Adapt pricing and operations to handle market shifts | `0.5 × satisfation + 0.5 × brand_reputation` |


---

## 🔁 How to Reproduce Baseline Results

```bash
# 1. Clone the repository
git clone https://github.com/samrataditya074-hue/Building-a-Real-World-AI-Training-Environment-using-OpenEnv
cd Building-a-Real-World-AI-Training-Environment-using-OpenEnv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run deterministic baseline
python baseline_inference.py --seed 42

# Expected output (LLM policy, seed=42):
# {
#   "seed": 42,
#   "model": "llama-3.3-70b-versatile",
#   "scores": {"easy": 1.0, "medium": 0.94, "hard": 0.47},
#   "quarters_run": {"easy": 4, "medium": 50, "hard": 50}
# }
```

---

## 🙋 FAQ

**Q: I'm not technical — how do I try this?**
```bash
pip install -r requirements.txt
python demo_nontech.py
```
That's it. You'll see the AI making decisions in plain English, and rotating tips explaining what's happening.

**Q: How does the AI decide what to do?**
The AI uses a technique called Reinforcement Learning — it practiced running millions of virtual companies and learned which decisions lead to the best outcomes. It's like how a new manager learns from experience, but sped up 1,000,000x.

**Q: What happens if the AI makes bad decisions?**
It can go bankrupt (cash hits $0), lose all customers (satisfaction hits 0%), or have no employees left. These are "terminal failure" conditions. The AI tries to avoid them, but random world events (recessions, labor strikes) can derail even the best strategy.

**Q: Can I change the AI's decisions?**
Yes! Use the **Manual CEO Mode** tab in the dashboard. You control all 8 levers yourself and see the results live.

**Q: What is OpenEnv?**
OpenEnv is a framework for building AI training environments. It provides a standard way to define how an AI interacts with a simulated world — so researchers and developers can easily plug in different AI algorithms and compare performance.

---

## 🗂️ Project Structure

```text
open_env_project/
├── models.py              ← Data blueprints (Employee, Action, Observation, State, Reward)
├── graders.py             ← Three deterministic task graders (episode-history aware)
├── baseline_inference.py  ← Run all 3 tasks, output JSON scores (OpenAI or heuristic)
├── demo_nontech.py        ← Plain-English CEO story (non-technical demo)
├── train_rl.py            ← Train the AI with PPO (10M steps)
├── main.py                ← Temperature control env demo
├── requirements.txt       ← Python dependencies
├── Dockerfile             ← Container for deployment
├── openenv.yaml           ← Full OpenEnv spec (actions, observations, tasks)
├── CHANGELOG.md           ← What changed and why
│
├── env/                   ← Gymnasium environments (legacy/alternative)
│   ├── business_env.py    ← CEO sim (Gymnasium version)
│   ├── business_state.py  ← State for Gymnasium version
│   ├── control_env.py     ← Industrial temperature control
│   └── state.py           ← State for control env
│
├── agent/
│   ├── business_agent.py  ← CEO AI (RL model + hand-coded heuristic)
│   └── baseline.py        ← Simple older rule-based agent
│
├── server/
│   ├── app.py             ← Gradio dashboard + FastAPI + OpenEnv API
│   └── environment.py     ← Main CEO simulation (OpenEnv-compliant)
│
└── tests/
    ├── test_ceo_env.py         ← Core environment integration tests
    ├── test_openenv_spec.py    ← OpenEnv API compliance tests
    ├── test_graders.py         ← Deterministic grader tests
    └── test_reward_shaping.py  ← Reward signal quality tests
```

---

## 📐 Reward Function

The AI receives a score each quarter:

```
reward = pos - neg + profit_delta + morale_delta + rd_payoff + fire_penalty

pos = profit/80 + satisfaction/40 + morale/60 + reputation/80 + rd_progress/100
neg = losses/80 + low_morale_penalty + low_reputation_penalty + crises × 0.5

profit_delta  = (profit_this_q - profit_last_q) / 5000   # reward improvement
morale_delta  = (morale_this_q - morale_last_q) / 10     # reward team growth
rd_payoff     = rd_progress/100 × 0.2                    # cumulative R&D bonus
fire_penalty  = -0.3 × (people_fired / 5)                # punish mass layoffs

terminal penalty: -50 if company dies (discourages "suicide strategies")
```

---

## 🏗️ Schema Design

The environment strictly follows the OpenEnv v1 specification. See `openenv.yaml` for the complete machine-readable schema including all action/observation field descriptions, ranges, and task grader references.

### OpenEnv API
```python
from server.environment import CEOEnvironment
from models import Action

env = CEOEnvironment()
obs = env.reset(seed=42)        # → Observation
obs = env.step(Action(...))     # → Observation (with .reward, .done, .info)
info = env.state()              # → dict (OpenEnv spec)
typed = env.typed_state()       # → State (full internal state with history)
```

---

## 🛠️ Implementation Updates & Improvements

In our most recent update, we have implemented several major technical improvements to push this platform to a production-ready release:

1. **Full OpenEnv Compliance**: Rewrote the environment (`server/environment.py`) to fully comply with the OpenEnv specification, including robust `openenv.yaml` schemas to define all action spaces and observation states.
2. **Deterministic Task Graders**: Replaced ad-hoc end-of-episode evaluation with three episode-history-aware deterministic graders (`graders.py`) to consistently evaluate agents across Easy (Survive), Medium (Grow), and Hard (Win) tasks.
3. **Continuous Reward Shaping**: Restructured the reward signal, adding quarter-over-quarter metrics (profit delta, morale delta) with continuous shaping to ensure stable Reinforcement Learning convergence.
4. **Pydantic Data Models**: Upgraded `Action` and `Observation` schemas to tightly validated `pydantic.BaseModel` components. This instantly blocks invalid agent actions during simulations.
5. **Testing & CI/CD**: Added comprehensive test suites running via `pytest` to validate environmental integrity, action rejection, and reward bounds checking.
