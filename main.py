import argparse
from env.control_env import IndustrialControlEnv
from agent.baseline import PIDController

def run_simulation(difficulty: str, steps: int = 100):
    print(f"--- Starting Industrial Control Simulation ---")
    print(f"Difficulty: {difficulty}")
    
    env = IndustrialControlEnv(difficulty=difficulty)
    agent = PIDController()
    
    obs, info = env.reset()
    total_reward = 0.0
    
    for i in range(steps):
        # The typed state can be retrieved as per OpenEnv requirement
        state = env.state()
        
        # Agent computes action based on current state parameters
        action = agent.compute_action(
            current_temp=state.current_temperature,
            target_temp=state.target_temperature
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1:03d} | Temp: {state.current_temperature:.2f} | Target: {state.target_temperature:.2f} | Action: {action[0]:+.2f} | Reward: {reward:+.2f}")
        
        if terminated or truncated:
            print("Simulation ended.")
            break
            
    print(f"Total Cumulative Reward: {total_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenEnv RL Simulation locally")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], help="Environment Difficulty")
    parser.add_argument("--steps", type=int, default=200, help="Number of maximum simulation steps")
    args = parser.parse_args()
    
    run_simulation(args.difficulty, args.steps)
