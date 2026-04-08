import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import logging

from server.environment import CEOEnvironment
from models import Action

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Track steps for controlled logging
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        np_obs = obs.to_array()
        return np_obs, {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # --- NEW: Utility function for safe RL output scaling ---
        def scale_positive(x: float) -> float:
            """Safely map RL outputs from [-1.0, 1.0] to [0.0, 1.0]."""
            return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0))

        scaled_mkt = scale_positive(action[1])
        scaled_rd = scale_positive(action[3])
        
        # Controlled logging: Log every 50,000 steps to avoid performance bottlenecks
        if self.step_count % 50000 == 0:
            logger.info(f"Env Step {self.step_count}: Scaling -> mkt={action[1]:.2f}->{scaled_mkt:.2f}, rd={action[3]:.2f}->{scaled_rd:.2f}")

        # Decode the raw numpy array into typed Action
        a = Action(
            price_adjustment=float(action[0]),
            marketing_push=scaled_mkt,
            hire_fire=float(action[2]),
            rd_investment=scaled_rd,
            salary_adjustment=float(action[4]),
            task_allocation=float(action[5]),
            crisis_response=float(action[6]),
            budget_shift=float(action[7])
        )
        
        obs = self.env.step(a)
        
        return obs.to_array(), obs.reward, obs.done, False, obs.info

def main():
    logger.info("Initializing RL Environment")
    vec_env = CEOEnvWrapper()

    logger.info("Instantiating PPO Agent")
    # Set verbose=0 to completely silence Stable Baselines 3 stdout per rollout
    model = PPO("MlpPolicy", vec_env, verbose=0)

    try:
        logger.info("Starting Training (10,000,000 steps). Press Ctrl+C to safely interrupt and save early.")
        model.learn(total_timesteps=10_000_000)
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C). Saving current progress...")
    finally:
        logger.info("Saving Model to 'ceo_ppo_model.zip'")
        model.save("ceo_ppo_model")
        logger.info("Agent saved successfully. Use this model for inference via PPO.load('ceo_ppo_model')")

if __name__ == "__main__":
    main()
