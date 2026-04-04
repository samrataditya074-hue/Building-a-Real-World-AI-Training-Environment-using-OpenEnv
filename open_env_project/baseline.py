import argparse
import random
import numpy as np

from server.environment import CEOEnvironment
from agent.business_agent import CorporateAgent
from tasks import TASKS
from models import Action

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def evaluate_task(env: CEOEnvironment, agent: CorporateAgent, task_name: str, scorer, max_steps: int = 100):
    obs = env.reset()
    for _ in range(max_steps):
        # The agent's current logic expects an array corresponding to Observation
        action_np = agent.compute_action(obs.to_array())
        
        # Mapping to the typed model
        action = Action(
            price_adjustment=float(action_np[0]),
            marketing_push=float(action_np[1]),
            hire_fire=float(action_np[2]),
            rd_investment=float(action_np[3]),
            salary_adjustment=float(action_np[4]),
            task_allocation=float(action_np[5]),
            crisis_response=float(action_np[6]),
            budget_shift=float(action_np[7])
        )
        
        obs = env.step(action)
        
        if obs.done:
            break
            
    final_score = scorer(env.state())
    return final_score

def main():
    parser = argparse.ArgumentParser(description="Baseline Agent Evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"--- Running Baseline Agent (Seed: {args.seed}) ---")
    set_seed(args.seed)

    env = CEOEnvironment()
    agent = CorporateAgent()
    
    for task_name, scorer in TASKS.items():
        set_seed(args.seed)  # Re-seed for consistent task runs independently
        score = evaluate_task(env, agent, task_name, scorer)
        print(f"Task '{task_name.capitalize()}': Score = {score:.2f}")

if __name__ == "__main__":
    main()
