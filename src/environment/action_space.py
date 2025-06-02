"""
Action Space Implementation
Defines the 7 discrete actions for the autonomous parking DQN agent.

Based on Table 1 from the research paper:
"Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"

Action Space (7 discrete actions):
ID | Symbol | Description      | Δδ (steer) | Δv (m/s)
---|--------|------------------|------------|----------
0  | a_0    | Hold (brake)     | 0°         | -0.6
1  | a_1    | Throttle forward | 0°         | +0.6  
2  | a_2    | Reverse back     | 0°         | -0.6
3  | a_3    | Left forward     | +8°        | +0.6
4  | a_4    | Right forward    | -8°        | +0.6
5  | a_5    | Left reverse     | +8°        | -0.6
6  | a_6    | Right reverse    | -8°        | -0.6
"""

import math
import numpy as np
from typing import Dict, Tuple, List
from enum import IntEnum


class ActionType(IntEnum):
    """Enumeration of the 7 discrete actions."""
    HOLD_BRAKE = 0      # a_0: Hold/Brake
    THROTTLE_FORWARD = 1 # a_1: Throttle forward
    REVERSE_BACK = 2     # a_2: Reverse back
    LEFT_FORWARD = 3     # a_3: Left forward
    RIGHT_FORWARD = 4    # a_4: Right forward
    LEFT_REVERSE = 5     # a_5: Left reverse
    RIGHT_REVERSE = 6    # a_6: Right reverse


class ActionSpace:
    """
    Manages the discrete action space for the parking environment.
    
    This class encapsulates the action definitions, parameter mappings,
    and action validation according to the research paper specifications.
    """
    
    def __init__(self):
        """Initialize action space with parameters from the paper."""
        
        # Action parameters: (velocity_change_m_per_s, steering_change_degrees)
        self.action_params: Dict[int, Tuple[float, float]] = {
            ActionType.HOLD_BRAKE:      (-0.6,  0.0),  # Brake/hold position
            ActionType.THROTTLE_FORWARD: (+0.6,  0.0),  # Forward throttle
            ActionType.REVERSE_BACK:     (-0.6,  0.0),  # Reverse throttle
            ActionType.LEFT_FORWARD:     (+0.6, +8.0),  # Left steering + forward
            ActionType.RIGHT_FORWARD:    (+0.6, -8.0),  # Right steering + forward
            ActionType.LEFT_REVERSE:     (-0.6, +8.0),  # Left steering + reverse
            ActionType.RIGHT_REVERSE:    (-0.6, -8.0),  # Right steering + reverse
        }
        
        # Action descriptions for logging/debugging
        self.action_descriptions: Dict[int, str] = {
            ActionType.HOLD_BRAKE:      "Hold/Brake",
            ActionType.THROTTLE_FORWARD: "Throttle Forward", 
            ActionType.REVERSE_BACK:     "Reverse Back",
            ActionType.LEFT_FORWARD:     "Left Forward",
            ActionType.RIGHT_FORWARD:    "Right Forward", 
            ActionType.LEFT_REVERSE:     "Left Reverse",
            ActionType.RIGHT_REVERSE:    "Right Reverse",
        }
        
        # Action space size
        self.n_actions = 7
        
        # Physics constraints
        self.max_velocity_change = 0.6  # m/s per timestep
        self.max_steering_change = 8.0  # degrees per timestep
        
    def get_action_params(self, action: int) -> Tuple[float, float]:
        """
        Get the velocity and steering changes for a given action.
        
        Args:
            action: Action ID (0-6)
            
        Returns:
            Tuple of (velocity_change_m_per_s, steering_change_degrees)
            
        Raises:
            ValueError: If action is invalid
        """
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}. Must be 0-{self.n_actions-1}")
            
        return self.action_params[action]
    
    def get_velocity_change(self, action: int) -> float:
        """Get velocity change in m/s for given action."""
        velocity_change, _ = self.get_action_params(action)
        return velocity_change
    
    def get_steering_change(self, action: int) -> float:
        """Get steering change in degrees for given action."""
        _, steering_change = self.get_action_params(action)
        return steering_change
    
    def get_steering_change_radians(self, action: int) -> float:
        """Get steering change in radians for given action."""
        return math.radians(self.get_steering_change(action))
    
    def get_action_description(self, action: int) -> str:
        """Get human-readable description of action."""
        if not self.is_valid_action(action):
            return f"Invalid Action ({action})"
        return self.action_descriptions[action]
    
    def is_valid_action(self, action: int) -> bool:
        """Check if action ID is valid."""
        return 0 <= action < self.n_actions
    
    def get_all_actions(self) -> List[int]:
        """Get list of all valid action IDs."""
        return list(range(self.n_actions))
    
    def get_action_summary(self) -> Dict[int, Dict[str, any]]:
        """
        Get complete summary of all actions.
        
        Returns:
            Dictionary mapping action ID to action details
        """
        summary = {}
        for action_id in self.get_all_actions():
            velocity_change, steering_change = self.get_action_params(action_id)
            summary[action_id] = {
                'description': self.get_action_description(action_id),
                'velocity_change_ms': velocity_change,
                'steering_change_deg': steering_change,
                'steering_change_rad': math.radians(steering_change),
                'type': ActionType(action_id).name
            }
        return summary
    
    def apply_action_to_car(self, car_agent, action: int, dt: float):
        """
        Apply action to a car agent with proper physics integration.
        
        Args:
            car_agent: CarAgent instance to modify
            action: Action ID to apply
            dt: Time step in seconds
        """
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}")
        
        velocity_change, steering_change_deg = self.get_action_params(action)
        
        # Apply velocity change (with time scaling)
        new_velocity = car_agent.velocity + (velocity_change * dt)
        
        # Apply steering change (with time scaling) 
        steering_change_rad = math.radians(steering_change_deg)
        new_steering = car_agent.steering_angle + (steering_change_rad * dt)
        
        # Apply limits and constraints
        car_agent.velocity = np.clip(new_velocity, car_agent.min_velocity, car_agent.max_velocity)
        car_agent.steering_angle = np.clip(new_steering, -car_agent.max_steering, car_agent.max_steering)
    
    def get_action_effects_matrix(self) -> np.ndarray:
        """
        Get matrix representation of action effects for analysis.
        
        Returns:
            Matrix of shape (n_actions, 2) where columns are [velocity_change, steering_change]
        """
        effects = np.zeros((self.n_actions, 2))
        for i in range(self.n_actions):
            velocity_change, steering_change = self.get_action_params(i)
            effects[i] = [velocity_change, steering_change]
        return effects
    
    def sample_random_action(self) -> int:
        """Sample a random valid action."""
        return np.random.randint(0, self.n_actions)
    
    def __str__(self) -> str:
        """String representation of action space."""
        lines = ["Action Space (7 discrete actions):"]
        lines.append("ID | Description      | Δv (m/s) | Δδ (deg)")
        lines.append("---|------------------|----------|----------")
        
        for action_id in self.get_all_actions():
            desc = self.get_action_description(action_id)
            vel_change, steer_change = self.get_action_params(action_id)
            lines.append(f"{action_id:2d} | {desc:16s} | {vel_change:+8.1f} | {steer_change:+8.1f}")
            
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representation of action space."""
        return f"ActionSpace(n_actions={self.n_actions})"


# Global instance for easy access
action_space = ActionSpace() 