import numpy as np

class PIDController:
    """
    Baseline Artificial Intelligence Agent (Proportional-Integral-Derivative Controller).
    A standard deterministic controller used in real-world control systems.
    """
    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_action_val = 0.0

    def compute_action(self, current_temp: float, target_temp: float) -> np.ndarray:
        error = target_temp - current_temp
        
        # Anti-windup for integral
        self.integral += error
        self.integral = max(min(self.integral, 20.0), -20.0) 
        
        derivative = error - self.prev_error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        
        # The environment expects an array with shape (1,)
        # Clip action within environment action space [-1.0, 1.0]
        action = max(min(output, 1.0), -1.0)
        self.last_action_val = float(action)
        return np.array([action], dtype=np.float32)
        
    def get_thought(self, current_temp: float, target_temp: float) -> str:
        """Translates math into a 'Thought Cloud' sentence."""
        diff = target_temp - current_temp
        if abs(diff) < 1.0:
            return "I'm just coasting now. Everything looks perfect! ✨"
        elif diff > 0:
            intensity = "Heating up gently." if self.last_action_val < 0.5 else "Heating aggressively!"
            return f"It's {diff:.1f}° too cold. {intensity} 🔥"
        else:
            intensity = "Cooling slightly." if self.last_action_val > -0.5 else "Cooling at full power!"
            return f"It's {abs(diff):.1f}° too hot. {intensity} ❄️"

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
