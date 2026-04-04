import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

from server.environment import CEOEnvironment
from models import Action

class CEOEnvWrapper(gym.Env):
    """
    Gymnasium wrapper for the OpenEnv CEOEnvironment so it can be 
    trained using standard RL libraries like Stable-Baselines3.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()
        self.env = CEOEnvironment()
        
        # Action space: 8 continuous levers, each strictly bound between -1.0 and 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # Observation space: 14 bounds [0.0, 2.0] generally (some flags are 0 or 1)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(14,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        np_obs = obs.to_array()
        return np_obs, {}

    def step(self, action: np.ndarray):
        # Decode the raw numpy array into typed Action
        a = Action(
            price_adjustment=float(action[0]),
            marketing_push=float(action[1]),
            hire_fire=float(action[2]),
            rd_investment=float(action[3]),
            salary_adjustment=float(action[4]),
            task_allocation=float(action[5]),
            crisis_response=float(action[6]),
            budget_shift=float(action[7])
        )
        
        obs = self.env.step(a)
        
        return obs.to_array(), obs.reward, obs.done, False, obs.info

def main():
    print("--- Initializing RL Environment ---")
    vec_env = CEOEnvWrapper()

    print("--- Instantiating PPO Agent ---")
    model = PPO("MlpPolicy", vec_env, verbose=1)

    print("--- Starting Training (10,000,000 steps) ---")
    model.learn(total_timesteps=10_000_000)

    print("--- Saving Model ---")
    model.save("ceo_ppo_model")
    print("Agent saved entirely to 'ceo_ppo_model.zip'. Use this model for inference loading via PPO.load()")

if __name__ == "__main__":
    main()
