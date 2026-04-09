import os
import json
import textwrap
import sys
from typing import List, Optional, Any
from openai import OpenAI

# Ensure project root is in path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Task Configuration Mapping
TASK_CONFIG = {
    "easy_revenue_target": {"name": "Revenue Growth Benchmark", "difficulty": "easy", "max_steps": 4, "grader": GRADERS["easy_revenue_target"]},
    "medium_budget_balance": {"name": "Budget Scaling & Workforce Management", "difficulty": "medium", "max_steps": 8, "grader": GRADERS["medium_budget_balance"]},
    "hard_strategic_growth": {"name": "Long-Term Valuation Optimization", "difficulty": "hard", "max_steps": 12, "grader": GRADERS["hard_strategic_growth"]},
}

# ──────────────────────────────────────────────────────────────────────────────
# STDOUT LOGGING HELPERS (Strictly following Hackathon Spec)
# ──────────────────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(task: str, name: str, difficulty: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] task={task} name=\"{name}\" difficulty={difficulty} success={str(success).lower()} steps={steps} score={score:.4f} rewards=[{rewards_str}]",
        flush=True
    )

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
# CORE EVALUATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def run_evaluation(client: OpenAI, env: CEOEnvironment, task_id: str) -> float:
    """Run a single task evaluation and return the final score."""
    if task_id not in TASK_CONFIG:
        return 0.0

    config = TASK_CONFIG[task_id]
    max_steps = config["max_steps"]
    grader_fn = config["grader"]
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(seed=42, task_id=task_id)
        
        for step in range(1, max_steps + 1):
            # Get observation in readable format for LLM
            obs_dict = {
                "cash": round(obs.cash_norm * 200000, 0),
                "revenue": round(obs.revenue_norm * 5000, 0),
                "morale": round(obs.employee_morale_norm * 100, 1),
                "satisfaction": round(obs.customer_satisfaction_norm * 100, 1),
                "reputation": round(obs.brand_reputation_norm * 100, 1),
                "crises": obs.info.get("crisis_count", 0)
            }
            
            action = get_model_action(client, obs_dict)
            action_desc = f"P:{action.price_adjustment:.1f},H:{action.hire_fire:.1f},R:{action.rd_investment:.1f}"
            
            result = env.step(action)
            reward = result.reward or 0.0
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=action_desc, reward=reward, done=done, error=None)
            if done: break
            obs = result

        # Calculate score using the task's specific grader
        history = env.state_obj.metrics_history
        score = grader_fn(history, seed=42)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
    finally:
        log_end(
            task=task_id, 
            name=config["name"], 
            difficulty=config["difficulty"],
            success=success, 
            steps=steps_taken, 
            score=score, 
            rewards=rewards
        )
    
    return score

# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CEOEnvironment()

    target_task = os.getenv("TASK_ID")
    
    if target_task and target_task in TASK_CONFIG:
        run_evaluation(client, env, target_task)
    else:
        # Fallback: Run all 3 tasks in sequence if no specific TASK_ID
        for task in TASK_CONFIG.keys():
            run_evaluation(client, env, task)

if __name__ == "__main__":
    main()
