"""
Main Parking Environment Class
OpenAI Gym-compatible 2D parking simulation environment.

Based on the methodology from the research paper:
"Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"

Phase 2 Integration: Action Space, Distance Sensors, and Reward System
"""

import gym
from gym import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Dict, Any, Optional

from .car_agent import CarAgent
from .renderer import ParkingRenderer
from .action_space import ActionSpace
from .sensors import SensorArray
from .rewards import RewardFunction


class ParkingEnv(gym.Env):
    """
    2D Parking Environment for DQN training.
    
    State Space: [x, y, θ, v, d_1, d_2, ..., d_8] (12 dimensions)
    Action Space: 7 discrete actions (hold, throttle, reverse, steer combinations)
    
    Phase 2 Features:
    - Modular action space with 7 discrete actions from paper
    - 8-directional distance sensors for obstacle detection
    - Comprehensive reward function with collision/success/progress/time components
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        width: float = 50.0,  # Environment width in meters
        height: float = 30.0,  # Environment height in meters
        dt: float = 0.1,      # Time step in seconds
        max_steps: int = 1000, # Maximum steps per episode
        render_mode: Optional[str] = None,
        sensor_max_range: float = 20.0,  # Maximum sensor range
        show_sensors: bool = True         # Whether to visualize sensors
    ):
        super(ParkingEnv, self).__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.show_sensors = show_sensors
        
        # Phase 2: Initialize modular components
        self.action_space_manager = ActionSpace()
        self.sensor_array = SensorArray(max_range=sensor_max_range)
        self.reward_function = RewardFunction()
        
        # State space: [x, y, θ, v, d_1, d_2, ..., d_8]
        # Position (x, y), orientation (θ), velocity (v), 8 distance sensors
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, -5.0] + [0.0] * 8),
            high=np.array([width, height, np.pi, 5.0] + [sensor_max_range] * 8),
            dtype=np.float32
        )
        
        # Action space: 7 discrete actions as per paper
        self.action_space = spaces.Discrete(self.action_space_manager.n_actions)
        
        # Car agent
        self.car = CarAgent(
            x=width * 0.1,  # Start near left side
            y=height * 0.5,  # Start in middle
            theta=0.0,       # Facing right initially
            velocity=0.0,
            wheelbase=2.5    # L = 2.5m as per paper
        )
        
        # Parking spot (target)
        self.target_x = width * 0.8  # Near right side
        self.target_y = height * 0.5  # Middle
        self.target_theta = 0.0       # Facing right
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        
        # Renderer
        self.renderer = None
        if render_mode:
            self.renderer = ParkingRenderer(width, height)
            
        # Environment boundaries for sensors
        self.environment_bounds = (0.0, 0.0, width, height)
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset car to random initial position (but not too close to target)
        while True:
            x = self.np_random.uniform(0.1 * self.width, 0.6 * self.width)
            y = self.np_random.uniform(0.1 * self.height, 0.9 * self.height)
            theta = self.np_random.uniform(-np.pi, np.pi)
            
            # Make sure starting position is not too close to target
            distance = math.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)
            if distance > 5.0:  # At least 5 meters away
                break
                
        self.car.reset(x, y, theta, 0.0)
        
        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        
        # Reset reward function progress tracking
        self.reward_function.reset_progress_tracking()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        
        # Validate action
        if not self.action_space_manager.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}. Must be 0-{self.action_space_manager.n_actions-1}")
        
        # Apply action to car using new action space system
        self.action_space_manager.apply_action_to_car(self.car, action, self.dt)
        
        # Update car physics
        self.car.update(self.dt)
        
        # Get sensor readings
        sensor_readings = self._get_distance_readings()
        
        # Check collision and boundary conditions
        is_collision = self._is_collision()
        is_out_of_bounds = self._is_out_of_bounds()
        
        # Calculate reward using new reward system
        reward_info = self.reward_function.calculate_reward(
            car_x=self.car.x,
            car_y=self.car.y,
            car_theta=self.car.theta,
            car_velocity=self.car.velocity,
            target_x=self.target_x,
            target_y=self.target_y,
            target_theta=self.target_theta,
            is_collision=is_collision,
            is_out_of_bounds=is_out_of_bounds,
            sensor_readings=sensor_readings,
            timestep=self.current_step,
            episode_done=False
        )
        
        reward = reward_info['total_reward']
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_done(is_collision, is_out_of_bounds, reward_info['is_successful'])
        
        # Get observation
        obs = self._get_observation()
        
        # Enhanced info dictionary with Phase 2 details
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'reward_components': reward_info['components'],
            'distance_to_target': reward_info['distance_to_target'],
            'angle_error': reward_info['angle_error'],
            'is_collision': is_collision,
            'is_out_of_bounds': is_out_of_bounds,
            'is_successful': reward_info['is_successful'],
            'action_description': self.action_space_manager.get_action_description(action),
            'sensor_readings': sensor_readings,
            'min_sensor_distance': min(sensor_readings),
            'collision_risk': self.sensor_array.detect_collision_risk(
                threshold_distance=2.0, 
                car_velocity=self.car.velocity
            )
        }
        
        self.current_step += 1
        
        return obs, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if self.renderer is None:
            self.renderer = ParkingRenderer(self.width, self.height)
        
        # Render basic environment
        result = self.renderer.render(
            self.car, self.target_x, self.target_y, self.target_theta, mode
        )
        
        # Add sensor visualization if enabled
        if self.show_sensors and hasattr(self.renderer, 'screen'):
            scale = self.renderer.scale
            self.sensor_array.visualize_sensors(
                self.renderer.screen,
                self.car.x, self.car.y, self.car.theta,
                scale=scale, show_rays=True
            )
        
        return result
    
    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation with Phase 2 sensor integration."""
        # Car state: [x, y, θ, v]
        car_state = [
            self.car.x,
            self.car.y,
            self.car.theta,
            self.car.velocity
        ]
        
        # Distance sensors: 8 directions using new sensor system
        distances = self._get_distance_readings()
        
        # Combine into state vector
        state = np.array(car_state + distances, dtype=np.float32)
        
        return state
    
    def _get_distance_readings(self) -> list:
        """
        Get distance sensor readings using the new sensor array system.
        """
        return self.sensor_array.get_all_readings(
            car_x=self.car.x,
            car_y=self.car.y,
            car_theta=self.car.theta,
            environment_bounds=self.environment_bounds,
            obstacles=None  # No obstacles in Phase 2, will be added in Phase 3
        )
    
    def _is_collision(self) -> bool:
        """
        Check for collision with environment boundaries.
        
        In Phase 2, we only check boundary collisions.
        Phase 3 will add obstacle collision detection.
        """
        # Get car corners for collision detection
        corners = self.car.get_corners()
        
        # Check if any corner is outside environment boundaries
        for corner in corners:
            x, y = corner
            if x < 0 or x > self.width or y < 0 or y > self.height:
                return True
        
        return False
    
    def _is_out_of_bounds(self) -> bool:
        """Check if car center is outside environment boundaries."""
        return (self.car.x < 0 or self.car.x > self.width or 
                self.car.y < 0 or self.car.y > self.height)
    
    def _is_done(self, is_collision: bool, is_out_of_bounds: bool, is_successful: bool) -> bool:
        """
        Check if episode should terminate.
        
        Episode ends on:
        - Collision (boundary or obstacles)
        - Successful parking
        - Maximum timesteps reached
        - Out of bounds
        """
        # Terminal conditions
        if is_collision or is_successful or is_out_of_bounds:
            return True
            
        # Timeout condition
        if self.current_step >= self.max_steps:
            return True
            
        return False
    
    # Additional Phase 2 methods for analysis and debugging
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get detailed information about the action space."""
        return {
            'n_actions': self.action_space_manager.n_actions,
            'action_summary': self.action_space_manager.get_action_summary(),
            'action_effects_matrix': self.action_space_manager.get_action_effects_matrix()
        }
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get detailed information about the sensor array."""
        return {
            'n_sensors': len(self.sensor_array.sensors),
            'max_range': self.sensor_array.max_range,
            'sensor_names': self.sensor_array.sensor_names,
            'last_readings': self.sensor_array.last_readings,
            'front_sensors': self.sensor_array.get_front_sensors(),
            'rear_sensors': self.sensor_array.get_rear_sensors(),
            'side_sensors': self.sensor_array.get_side_sensors(),
            'min_distance': self.sensor_array.get_minimum_distance()
        }
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get detailed information about the reward function."""
        return {
            'collision_penalty': self.reward_function.collision_penalty,
            'success_reward': self.reward_function.success_reward,
            'position_tolerance': self.reward_function.position_tolerance,
            'orientation_tolerance_deg': math.degrees(self.reward_function.orientation_tolerance),
            'statistics': self.reward_function.get_reward_statistics(last_n_episodes=100)
        }
    
    def test_all_actions(self) -> Dict[int, str]:
        """Test all actions and return their descriptions (for debugging)."""
        action_tests = {}
        for action_id in self.action_space_manager.get_all_actions():
            description = self.action_space_manager.get_action_description(action_id)
            velocity_change, steering_change = self.action_space_manager.get_action_params(action_id)
            action_tests[action_id] = f"{description}: Δv={velocity_change:+.1f}, Δδ={steering_change:+.1f}°"
        return action_tests
    
    def get_current_state_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of current state (for debugging)."""
        sensor_readings = self._get_distance_readings()
        
        return {
            'car_state': {
                'position': (self.car.x, self.car.y),
                'orientation_deg': math.degrees(self.car.theta),
                'velocity': self.car.velocity,
                'steering_angle_deg': math.degrees(self.car.steering_angle)
            },
            'target_state': {
                'position': (self.target_x, self.target_y),
                'orientation_deg': math.degrees(self.target_theta)
            },
            'distances': {
                'to_target': math.sqrt((self.car.x - self.target_x)**2 + (self.car.y - self.target_y)**2),
                'sensors': sensor_readings,
                'min_sensor': min(sensor_readings)
            },
            'status': {
                'collision': self._is_collision(),
                'out_of_bounds': self._is_out_of_bounds(),
                'successful_parking': self.reward_function._is_successful_parking(
                    self.car.x, self.car.y, self.car.theta,
                    self.target_x, self.target_y, self.target_theta
                )
            },
            'episode': {
                'step': self.current_step,
                'max_steps': self.max_steps,
                'total_reward': self.total_reward
            }
        }
        
    def __str__(self) -> str:
        """String representation of environment."""
        return (f"ParkingEnv(size={self.width}×{self.height}m, "
                f"max_steps={self.max_steps}, "
                f"n_actions={self.action_space_manager.n_actions}, "
                f"n_sensors={len(self.sensor_array.sensors)})")
    
    def __repr__(self) -> str:
        """Detailed representation of environment."""
        return (f"ParkingEnv(width={self.width}, height={self.height}, "
                f"dt={self.dt}, max_steps={self.max_steps}, "
                f"sensor_range={self.sensor_array.max_range})") 