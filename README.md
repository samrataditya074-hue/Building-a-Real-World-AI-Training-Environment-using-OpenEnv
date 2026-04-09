# 🏢 Autonomous CEO Simulator — OpenEnv

**A Production-Grade Corporate Ecosystem for Training Agentic AI**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/open-env)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Adityakunwar1207/autonomous-ceo-simulator)

## 🌟 Overview
The **Autonomous CEO Simulator** is a high-fidelity Reinforcement Learning environment designed for the Scaler Meta PyTorch Hackathon. It simulates the complex, partially observable lifecycle of a corporate entity, where an AI agent must master strategic decision-making across 8 distinct levers (Pricing, HR, R&D, etc.) to ensure company survival and valuation growth.

---

## 🚀 Key Features
- **Real-World Task Simulation**: Models a corporate fiscal lifecycle (not a game/toy).
- **Full OpenEnv Compliance**: Implements Pydantic typed models for Observation, Action, and Reward.
- **Densely Annotated Rewards**: Provides partial progress signals and penalizes destructive behavior (e.g., bankruptcy, mass resignations).
- **3-Tier Graded Evaluation**: Includes mandatory tasks: Survival (Easy), Valuation (Medium), and Pricing War (Hard).
- **WOW Factor Dashboard**: A real-time 3D Gradio UI with live LLM narration of the CEO's "thoughts."

---

## 🎛️ Action Space (8 Continuous Levers)
| Index | Name | Range | Description |
|---|---|---|---|
| 0 | `price_adjustment` | `[-1, 1]` | Adjusts product price ($±8) relative to competitors. |
| 1 | `marketing_push` | `[0, 1]` | Quarterly marketing spend ($0–600). |
| 2 | `hire_fire` | `[-1, 1]` | Workforce change (Hire/Fire 5 people). |
| 3 | `rd_investment` | `[0, 1]` | R&D investment ($0–400). |
| 4 | `salary_adjustment` | `[-1, 1]` | Company-wide salary changes. |
| 5 | `task_allocation` | `[-1, 1]` | Employee Specialization vs. Cross-training. |
| 6 | `crisis_response` | `[-1, 1]` | Investment vs. Cost-cutting during crises. |
| 7 | `budget_shift` | `[-1, 1]` | Sales Expansion vs. Operations Savings. |

---

## 📊 Observation Space (14 Normalized Features)
The environment returns a normalized vector including:
- **Financials**: Cash, Revenue, Debt.
- **Sentiments**: Employee Morale, Customer Satisfaction.
- **Operations**: Inventory, R&D Progress, Efficiency.
- **Market**: Economic Trend, Competitor Price.

---

## 🏆 Mandatory Tasks (Graders)
1. **Survival (`review_annual_report`)** [Easy]: Run the company for 4 quarters without bankruptcy.
2. **Valuation (`allocate_budget`)** [Medium]: Grow valuation by 20% over 50 quarters while keeping Morale > 60%.
3. **Pricing War (`negotiate_merger`)** [Hard]: Undercut competitors in 6 of 8 target quarters while remaining profitable.

---

## 🛠️ Setup & Usage

### 1. Local Development
```bash
pip install -r requirements.txt
python app.py
```

### 2. Docker Deployment
```bash
docker build -t ceo-sim .
docker run -p 7860:7860 -e HF_TOKEN=your_token_here ceo-sim
```

### 3. Baseline Inference
To reproduce the baseline scores:
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
python inference.py
```

---

## 📝 Compliance Notes
- **Log Format**: Strictly follows `[START]`, `[STEP]`, and `[END]` stdout markers.
- **Endpoints**: `/reset`, `/step`, and `/state` are fully exposed via FastAPI.
- **Grader Registry**: Tasks are mapped to `"graders:function"` for OpenEnv discovery.

---
**Developed for the Scaler Meta PyTorch Hackathon (2026)**
