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
# Default task matches the "Medium" (Valuation) task mandated by hackathon sample context
TASK_NAME = os.getenv("TASK_ID") or "allocate_budget"
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# ──────────────────────────────────────────────────────────────────────────────
# Task Configuration Mapping
# ──────────────────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "review_annual_report": {"max_steps": 4, "grader": GRADERS["review_annual_report"]},
    "allocate_budget":      {"max_steps": 50, "grader": GRADERS["allocate_budget"]},
    "negotiate_merger":    {"max_steps": 50, "grader": GRADERS["negotiate_merger"]},
}

# Fallback if unknown task provided
if TASK_NAME not in TASK_CONFIG:
    TASK_NAME = "allocate_budget"

MAX_STEPS = TASK_CONFIG[TASK_NAME]["max_steps"]
GRADER_FN = TASK_CONFIG[TASK_NAME]["grader"]

# ──────────────────────────────────────────────────────────────────────────────
# STDOUT LOGGING HELPERS (Strictly following Mandatory Hackathon Spec)
# ──────────────────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    # [START] task=<task_name> env=<benchmark> model=<model_name>
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

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
        # Reset with the specific task_id to ensure environment state is initialized correctly
        obs = env.reset(seed=42, task_id=TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            # Formulate observation for LLM
            obs_array = obs.to_array()
            obs_labels = [
                "cash_norm", "revenue_norm", "customer_satisfaction_norm",
                "employee_morale_norm", "inventory_norm", "market_trend",
                "total_employees_norm", "brand_reputation_norm",
                "operational_efficiency_norm", "rd_progress_norm",
                "debt_norm", "cash_crisis_flag", "morale_crisis_flag",
                "competitor_price_norm",
            ]
            obs_dict = {label: round(float(val), 3) for label, val in zip(obs_labels, obs_array)}
            
            action = get_model_action(client, obs_dict)
            
            # Record action string for logging
            action_desc = f"P:{action.price_adjustment:.1f},M:{action.marketing_push:.1f},H:{action.hire_fire:.1f}"
            
            result = env.step(action)
            obs = result
            
            reward = result.reward or 0.0
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            
            # [STEP] log
            log_step(step=step, action=action_desc, reward=reward, done=done, error=None)

            if done:
                break

        # Calculate score using the task's specific grader
        history = env.typed_state().metrics_history
        score = GRADER_FN(history, seed=42)
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # Critical: [END] log must be emitted even on exception
        pass
    finally:
        # [END] log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
