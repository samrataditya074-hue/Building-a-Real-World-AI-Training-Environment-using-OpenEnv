import os
import sys
import yaml
import numpy as np
from server.environment import CEOEnvironment
from models import Action

def evaluate():
    """
    Evaluation script to run each task's canonical actions and print the results.
    Checks that graders are working and returning clamped scores.
    """
    # Load openenv.yaml to get tasks
    with open("openenv.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tasks = config.get("tasks", [])
    print(f"Starting Evaluation of {len(tasks)} tasks...\n")
    
    env = CEOEnvironment()
    
    for task in tasks:
        task_id = task.get("id")
        name = task.get("name")
        max_steps = task.get("max_steps", 1)
        
        print(f"--- Task: {name} ({task_id}) ---")
        
        # Reset with task_id
        obs = env.reset(seed=42, task_id=task_id)
        
        total_reward = 0
        done = False
        
        # Run a simple sequence of actions
        for i in range(max_steps):
            # Canonical action: price neutral, 50% marketing, 50% R&D, etc.
            action = Action(
                price_adjustment=0.0,
                marketing_push=0.5,
                hire_fire=0.0,
                rd_investment=0.5,
                salary_adjustment=0.0,
                task_allocation=0.0,
                crisis_response=0.1,
                budget_shift=0.1
            )
            obs = env.step(action)
            total_reward += obs.reward
            if obs.done:
                break
        
        # Get final score from rubric
        # In this env, episodic graders are used via env.rubric
        final_score = env.rubric.score_history(env.state_obj.metrics_history)
        
        print(f"Steps Run: {env.state_obj.quarter}")
        print(f"Total Step Reward: {total_reward:.4f}")
        print(f"Grader Score: {final_score:.4f}")
        
        if 0.0 <= final_score <= 1.0:
            print("OK: Score safely clamped within [0, 1]")
        else:
            print("ERROR: Score outside of [0, 1] range!")
        print("-" * 40)


if __name__ == "__main__":
    evaluate()
