import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

# Load local .env just for local testing/defaults
load_dotenv()

from openai import OpenAI
from server.environment import CEOEnvironment
from models import Action
from graders import GRADERS

# ─── Configuration ────────────────────────────────────────────────────────────
# Judges will provide these. Defaults point to our verified Groq configuration.
API_KEY = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") 
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile"

# Task configuration (must match openenv.yaml)
TASK_ID = os.getenv("TASK_ID") or "grow_val_medium"
BENCHMARK = "autonomous_ceo"
TEMPERATURE = 0.0  # Ensure deterministic strategic decisions

TASK_CONFIG = {
    "survive_easy": {"max_quarters": 4, "grader": GRADERS["survive_easy"]},
    "grow_val_medium": {"max_quarters": 50, "grader": GRADERS["grow_val_medium"]},
    "undercut_hard": {"max_quarters": 50, "grader": GRADERS["undercut_hard"]},
}

if TASK_ID not in TASK_CONFIG:
    print(f"Warning: Unknown TASK_ID={TASK_ID}. Defaulting to grow_val_medium.")
    TASK_ID = "grow_val_medium"

MAX_QUARTERS = TASK_CONFIG[TASK_ID]["max_quarters"]
GRADER_FN = TASK_CONFIG[TASK_ID]["grader"]

# ─── Logging Helpers ──────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Log format: [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Log format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── LLM Decision Logic ───────────────────────────────────────────────────────
def get_llm_action(client: OpenAI, obs_array: List[float]) -> Action:
    """Ask the LLM to decide the next CEO action based on company state."""
    obs_labels = [
        "cash_norm", "revenue_norm", "customer_satisfaction_norm",
        "employee_morale_norm", "inventory_norm", "market_trend",
        "total_employees_norm", "brand_reputation_norm",
        "operational_efficiency_norm", "rd_progress_norm",
        "debt_norm", "cash_crisis_flag", "morale_crisis_flag",
        "competitor_price_norm",
    ]
    obs_dict = {label: round(float(val), 3) for label, val in zip(obs_labels, obs_array)}

    system_prompt = (
        "You are an expert AI CEO. You receive the current company state as JSON. "
        "Reply ONLY with a JSON object containing these 8 keys, "
        "each value a float in [-1.0, +1.0] (marketing_push and rd_investment must be in [0.0, +1.0]): "
        "price_adjustment, marketing_push, hire_fire, rd_investment, "
        "salary_adjustment, task_allocation, crisis_response, budget_shift."
    )
    user_prompt = f"Current company state:\n{json.dumps(obs_dict, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
        )
        data = json.loads(response.choices[0].message.content)
        # Validate and clamp outputs
        import numpy as np
        return Action(
            price_adjustment=float(np.clip(data.get("price_adjustment", 0.0), -1.0, 1.0)),
            marketing_push=float(np.clip(data.get("marketing_push", 0.0), 0.0, 1.0)),
            hire_fire=float(np.clip(data.get("hire_fire", 0.0), -1.0, 1.0)),
            rd_investment=float(np.clip(data.get("rd_investment", 0.0), 0.0, 1.0)),
            salary_adjustment=float(np.clip(data.get("salary_adjustment", 0.0), -1.0, 1.0)),
            task_allocation=float(np.clip(data.get("task_allocation", 0.0), -1.0, 1.0)),
            crisis_response=float(np.clip(data.get("crisis_response", 0.0), -1.0, 1.0)),
            budget_shift=float(np.clip(data.get("budget_shift", 0.0), -1.0, 1.0)),
        )
    except Exception as exc:
        # On failure, return a neutral action (Action defaults all to 0.0)
        return Action()

# ─── Main Simulation Loop ─────────────────────────────────────────────────────
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CEOEnvironment()
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        obs = env.reset(seed=42)
        
        for q in range(1, MAX_QUARTERS + 1):
            # 1. Get LLM decision
            action = get_llm_action(client, obs.to_array().tolist())
            
            # 2. Step environment
            obs = env.step(action)
            
            # 3. Record metrics
            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = q
            
            # 4. Log step with strict format
            # action_str is just a summary for the logs
            action_desc = f"P:{action.price_adjustment:.1f}|M:{action.marketing_push:.1f}|H:{action.hire_fire:.1f}"
            log_step(step=q, action_str=action_desc, reward=reward, done=done, error=None)
            
            if done:
                break
        
        # 5. Final scoring using the correct task grader
        history = env.typed_state().metrics_history
        score = GRADER_FN(history, seed=42)
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        
        # Assuming 0.5 is the success threshold for evaluation
        success = score >= 0.5
        
    except Exception as e:
        # Ensure we still log something if the script crashes
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
