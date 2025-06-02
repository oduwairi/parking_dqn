"""
Main Parking Environment Class
OpenAI Gym-compatible 2D parking simulation environment.

Based on the methodology from the research paper:
"Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"
"""

import gym
from gym import spaces
import numpy as np
import pygame
import math
from typing import Tuple, Dict, Any, Optional

from .car_agent import CarAgent
from .renderer import ParkingRenderer


class ParkingEnv(gym.Env):
    """
    2D Parking Environment for DQN training.
    
    State Space: [x, y, θ, v, d_1, d_2, ..., d_8] (12 dimensions)
    Action Space: 7 discrete actions (hold, throttle, reverse, steer combinations)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        width: float = 50.0,  # Environment width in meters
        height: float = 30.0,  # Environment height in meters
        dt: float = 0.1,      # Time step in seconds
        max_steps: int = 1000, # Maximum steps per episode
        render_mode: Optional[str] = None
    ):
        super(ParkingEnv, self).__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # State space: [x, y, θ, v, d_1, d_2, ..., d_8]
        # Position (x, y), orientation (θ), velocity (v), 8 distance sensors
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, -5.0] + [-1.0] * 8),
            high=np.array([width, height, np.pi, 5.0] + [50.0] * 8),
            dtype=np.float32
        )
        
        # Action space: 7 discrete actions as per paper
        self.action_space = spaces.Discrete(7)
        
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
        
        # Tolerances for successful parking (from paper)
        self.position_tolerance = 0.5  # ε_p = 0.5m
        self.orientation_tolerance = math.radians(10)  # ε_θ = 10°
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        
        # Renderer
        self.renderer = None
        if render_mode:
            self.renderer = ParkingRenderer(width, height)
            
        # For tracking progress reward
        self.last_distance_to_target = None
        
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
        self.last_distance_to_target = self._distance_to_target()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Apply action to car
        self.car.apply_action(action, self.dt)
        
        # Update car physics
        self.car.update(self.dt)
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Get observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'distance_to_target': self._distance_to_target(),
            'is_collision': self._is_collision(),
            'is_successful': self._is_successful_parking(),
        }
        
        self.current_step += 1
        
        return obs, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if self.renderer is None:
            self.renderer = ParkingRenderer(self.width, self.height)
        
        return self.renderer.render(self.car, self.target_x, self.target_y, self.target_theta, mode)
    
    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Car state: [x, y, θ, v]
        car_state = [
            self.car.x,
            self.car.y,
            self.car.theta,
            self.car.velocity
        ]
        
        # Distance sensors: 8 directions
        distances = self._get_distance_readings()
        
        # Combine into state vector
        state = np.array(car_state + distances, dtype=np.float32)
        
        return state
    
    def _get_distance_readings(self) -> list:
        """
        Get distance sensor readings in 8 directions.
        Simulates ultrasonic/LiDAR sensors around the car.
        """
        distances = []
        
        # 8 sensor directions (every 45 degrees)
        sensor_angles = [i * math.pi / 4 for i in range(8)]
        
        for angle in sensor_angles:
            # Convert to global angle
            global_angle = self.car.theta + angle
            
            # Cast ray from car position
            distance = self._cast_ray(
                self.car.x, 
                self.car.y, 
                global_angle, 
                max_distance=20.0
            )
            distances.append(distance)
        
        return distances
    
    def _cast_ray(self, x: float, y: float, angle: float, max_distance: float) -> float:
        """
        Cast a ray from (x, y) in direction 'angle' and return distance to obstacle.
        For Phase 1, only check environment boundaries.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Check distance to environment boundaries
        distances = []
        
        # Distance to right boundary
        if dx > 0:
            distances.append((self.width - x) / dx)
        
        # Distance to left boundary  
        if dx < 0:
            distances.append(-x / dx)
            
        # Distance to top boundary
        if dy > 0:
            distances.append((self.height - y) / dy)
            
        # Distance to bottom boundary
        if dy < 0:
            distances.append(-y / dy)
            
        # Return minimum distance (closest boundary) but cap at max_distance
        min_distance = min(distances) if distances else max_distance
        return min(min_distance, max_distance)
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on paper methodology:
        - Collision penalty: -100 (episode termination)
        - Success reward: +100 (episode termination) 
        - Progress reward: +1 (closer), -0.5 (further)
        - Time penalty: -0.1 (per timestep)
        """
        reward = 0.0
        
        # Time penalty (encourage efficiency)
        reward -= 0.1
        
        # Check for collision
        if self._is_collision():
            reward -= 100.0
            return reward
        
        # Check for successful parking
        if self._is_successful_parking():
            reward += 100.0
            return reward
        
        # Progress reward
        current_distance = self._distance_to_target()
        if self.last_distance_to_target is not None:
            if current_distance < self.last_distance_to_target:
                reward += 1.0  # Getting closer
            else:
                reward -= 0.5  # Getting further
        
        self.last_distance_to_target = current_distance
        
        return reward
    
    def _distance_to_target(self) -> float:
        """Calculate Euclidean distance to target parking spot."""
        return math.sqrt(
            (self.car.x - self.target_x)**2 + 
            (self.car.y - self.target_y)**2
        )
    
    def _is_collision(self) -> bool:
        """Check if car has collided with environment boundaries."""
        # Car dimensions (simplified as point for Phase 1)
        margin = 1.0  # 1 meter safety margin
        
        return (
            self.car.x < margin or 
            self.car.x > self.width - margin or
            self.car.y < margin or 
            self.car.y > self.height - margin
        )
    
    def _is_successful_parking(self) -> bool:
        """Check if car is successfully parked within tolerances."""
        distance_ok = self._distance_to_target() <= self.position_tolerance
        
        # Normalize angle difference
        angle_diff = abs(self.car.theta - self.target_theta)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
        orientation_ok = angle_diff <= self.orientation_tolerance
        
        # Car should be nearly stationary when parked
        velocity_ok = abs(self.car.velocity) < 0.5
        
        return distance_ok and orientation_ok and velocity_ok
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        return (
            self.current_step >= self.max_steps or
            self._is_collision() or
            self._is_successful_parking()
        ) 