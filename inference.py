import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from server.environment import CEOEnvironment
from models import Action
from graders import GRADERS

# ──────────────────────────────────────────────────────────────────────────────
# MANDATORY CONFIGURATION (Scaler Hackathon)
# ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Benchmark and Task metadata
BENCHMARK = "Autonomous CEO Simulator"
TASK_NAME = os.getenv("TASK_ID") or "allocate_budget" # Default task
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Task Configuration Mapping
TASK_CONFIG = {
    "review_annual_report": {"max_steps": 3, "grader": GRADERS["review_annual_report"]},
    "allocate_budget": {"max_steps": 4, "grader": GRADERS["allocate_budget"]},
    "negotiate_merger": {"max_steps": 5, "grader": GRADERS["negotiate_merger"]},
    "evaluate_market_strategy": {"max_steps": 8, "grader": GRADERS["evaluate_market_strategy"]},
}

# Fallback if unknown task provided
if TASK_NAME not in TASK_CONFIG:
    TASK_NAME = "allocate_budget"

MAX_STEPS = TASK_CONFIG[TASK_NAME]["max_steps"]
GRADER_FN = TASK_CONFIG[TASK_NAME]["grader"]

# ──────────────────────────────────────────────────────────────────────────────
# STDOUT LOGGING HELPERS (Strictly following Hackathon Spec)
# ──────────────────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL INTERACTION
# ──────────────────────────────────────────────────────────────────────────────
def get_model_action(client: OpenAI, obs_dict: dict) -> Action:
    """Ask the LLM to decide the next CEO action based on company state."""
    system_prompt = (
        "You are an expert AI CEO managing a corporate simulation. "
        "Your goal is to maximize company valuation and operational success. "
        "Reply ONLY with a JSON object containing these 8 keys (floats in [-1.0, 1.0]): "
        "price_adjustment, marketing_push, hire_fire, rd_investment, "
        "salary_adjustment, task_allocation, crisis_response, budget_shift. "
        "Note: marketing_push and rd_investment must be in [0.0, 1.0]."
    )
    user_prompt = f"Current Business Metrics (JSON):\n{json.dumps(obs_dict, indent=2)}\nDecision:"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        data = json.loads(completion.choices[0].message.content)
        return Action(
            price_adjustment=float(data.get("price_adjustment", 0.0)),
            marketing_push=float(data.get("marketing_push", 0.0)),
            hire_fire=float(data.get("hire_fire", 0.0)),
            rd_investment=float(data.get("rd_investment", 0.0)),
            salary_adjustment=float(data.get("salary_adjustment", 0.0)),
            task_allocation=float(data.get("task_allocation", 0.0)),
            crisis_response=float(data.get("crisis_response", 0.0)),
            budget_shift=float(data.get("budget_shift", 0.0)),
        )
    except Exception as e:
        # Fallback to neutral action
        return Action()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CEOEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(seed=42, task_id=TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            # Get observation in readable format for LLM
            obs_dict = {
                "cash": round(obs.cash_norm * 200000, 0),
                "revenue": round(obs.revenue_norm * 5000, 0),
                "morale": round(obs.employee_morale_norm * 100, 1),
                "satisfaction": round(obs.customer_satisfaction_norm * 100, 1),
                "market_trend": obs.market_trend,
                "reputation": round(obs.brand_reputation_norm * 100, 1),
                "active_crisis": bool(obs.cash_crisis_flag or obs.morale_crisis_flag)
            }
            
            action = get_model_action(client, obs_dict)
            
            # Record action description for logging
            action_desc = f"Pricing:{action.price_adjustment:.1f},Hire:{action.hire_fire:.1f},R&D:{action.rd_investment:.1f}"
            
            result = env.step(action)
            obs = result
            
            reward = result.reward or 0.0
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_desc, reward=reward, done=done, error=None)

            if done:
                break

        # Calculate score using the task's specific grader
        history = env.state_obj.metrics_history
        score = GRADER_FN(history, seed=42)
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # Final log must happen even on failure
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
