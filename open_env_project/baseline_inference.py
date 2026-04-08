"""
baseline_inference.py — Deterministic baseline evaluation across all 3 OpenEnv tasks.

Runs the CEO simulation with either:
  (A) An OpenAI LLM policy (if OPENAI_API_KEY is set)
  (B) The built-in CorporateAgent heuristic (fallback, always available)

Outputs a JSON summary with scores for all 3 tasks.

Usage:
    python baseline_inference.py --seed 42
    python baseline_inference.py --seed 42 --openai-model gpt-4o-mini
    OPENAI_API_KEY=sk-... python baseline_inference.py --seed 42

Expected output (heuristic policy, seed=42):
    {
      "seed": 42,
      "model": "heuristic",
      "scores": {"easy": 1.0, "medium": 0.71, "hard": 0.52},
      "quarters_run": {"easy": 4, "medium": 50, "hard": 50}
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, Any

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import CEOEnvironment
from agent.business_agent import CorporateAgent
from models import Action
from graders import GRADERS

# ── Optional OpenAI import ────────────────────────────────────────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ──────────────────────────────────────────────────────────────────────────────
# Task configuration
# max_quarters controls how long each task episode runs
# ──────────────────────────────────────────────────────────────────────────────
TASK_CONFIG: Dict[str, int] = {
    "easy": 4,    # Survive 4 quarters
    "medium": 50, # Grow valuation over 50 quarters
    "hard": 50,   # Win pricing war over 50 quarters (first 8 evaluated)
}


def _set_seed(seed: int) -> None:
    """Seed all RNGs for fully reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)

# --- NEW: Utility function for safe RL output scaling ---
def scale_positive(x: float) -> float:
    """Safely map RL outputs from [-1.0, 1.0] to [0.0, 1.0] bounds."""
    return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))

def _heuristic_action(obs_array: np.ndarray, agent: CorporateAgent) -> Action:
    """Convert heuristic agent output to typed Action."""
    action_np = agent.compute_action(obs_array)
    return Action(
        price_adjustment=float(action_np[0]),
        marketing_push=scale_positive(action_np[1]), # mapped [-1, 1] -> [0, 1]
        hire_fire=float(action_np[2]),
        rd_investment=scale_positive(action_np[3]),  # mapped [-1, 1] -> [0, 1]
        salary_adjustment=float(action_np[4]),
        task_allocation=float(action_np[5]),
        crisis_response=float(action_np[6]),
        budget_shift=float(action_np[7]),
    )


def _llm_action(obs_array: np.ndarray, client: "OpenAI", model: str) -> Action:
    """
    Ask an OpenAI LLM to decide the next CEO action.

    The observation is sent as a structured JSON prompt.
    The LLM responds with 8 floats in [-1, +1].
    Falls back to neutral action if parsing fails.
    """
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
        "You are an expert AI CEO making quarterly business decisions. "
        "You receive the current company state as JSON. "
        "Reply ONLY with a JSON object containing exactly these 8 keys, "
        "each value a float in [-1.0, +1.0] (marketing_push and rd_investment "
        "must be in [0.0, +1.0]): "
        "price_adjustment, marketing_push, hire_fire, rd_investment, "
        "salary_adjustment, task_allocation, crisis_response, budget_shift."
    )
    user_prompt = f"Current company state:\n{json.dumps(obs_dict, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # deterministic
        )
        data = json.loads(response.choices[0].message.content)
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
        print(f"  [LLM parse error, using neutral action]: {exc}", file=sys.stderr)
        return Action()


def run_task(
    task_name: str,
    seed: int,
    use_llm: bool,
    openai_client: Any,
    openai_model: str,
) -> Dict[str, Any]:
    """
    Run one full task episode and return its score + metadata.

    Args:
        task_name:     "easy", "medium", or "hard"
        seed:          RNG seed for reproducibility
        use_llm:       If True, use OpenAI API for decisions
        openai_client: OpenAI client (or None)
        openai_model:  Model name string

    Returns:
        Dict with keys: task, seed, quarters_run, score
    """
    max_q = TASK_CONFIG[task_name]
    grader_fn = GRADERS[task_name]

    # Fresh environment + agent seeded identically for each task
    _set_seed(seed)
    env = CEOEnvironment()
    agent = CorporateAgent()
    obs = env.reset(seed=seed)

    print(f"  Running task '{task_name}' for up to {max_q} quarters …", flush=True)

    for q in range(max_q):
        obs_array = obs.to_array()

        if use_llm and openai_client is not None:
            action = _llm_action(obs_array, openai_client, openai_model)
        else:
            action = _heuristic_action(obs_array, agent)

        obs = env.step(action)

        if obs.done:
            print(f"    Episode ended at Q{q + 1}", flush=True)
            break

    # Use typed_state() to access metrics_history for the grader
    history = env.typed_state().metrics_history
    score = grader_fn(history, seed=seed)

    print(f"  Score: {score:.4f}  (quarters completed: {len(history)})")
    return {
        "task": task_name,
        "seed": seed,
        "quarters_run": len(history),
        "score": round(score, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic baseline evaluation across all 3 OpenEnv tasks."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--openai-model", type=str, default="gpt-4o-mini",
        help="OpenAI model to use if OPENAI_API_KEY is set (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional: path to write JSON output file"
    )
    args = parser.parse_args()

    # Determine which policy to use
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_llm = HAS_OPENAI and bool(api_key)
    openai_client = OpenAI(api_key=api_key) if use_llm else None
    model_name = args.openai_model if use_llm else "heuristic"

    print("=" * 60)
    print(f"Autonomous CEO — Baseline Inference")
    print(f"  Seed  : {args.seed}")
    print(f"  Policy: {model_name}")
    print("=" * 60)

    results: Dict[str, Any] = {
        "seed": args.seed,
        "model": model_name,
        "scores": {},
        "quarters_run": {},
    }

    for task in ["easy", "medium", "hard"]:
        print(f"\n[Task: {task.upper()}]")
        task_result = run_task(
            task_name=task,
            seed=args.seed,
            use_llm=use_llm,
            openai_client=openai_client,
            openai_model=args.openai_model,
        )
        results["scores"][task] = task_result["score"]
        results["quarters_run"][task] = task_result["quarters_run"]

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(json.dumps(results, indent=2))
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
