"""
Main Parking Environment Class - Enhanced for Phase 3
OpenAI Gym-compatible 2D parking simulation environment.

Based on the methodology from the research paper:
"Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"

Phase 3 Integration: Static Obstacles, Collision Detection, and Enhanced Reward Engineering
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
from .obstacles import ObstacleManager
from .parking_spots import ParkingSpotManager
from .collision_detection import CollisionDetector


class ParkingEnv(gym.Env):
    """
    Enhanced 2D Parking Environment for DQN training with Phase 3 features.
    
    State Space: [x, y, θ, v, d_1, d_2, ..., d_8] (12 dimensions)
    Action Space: 7 discrete actions (hold, throttle, reverse, steer combinations)
    
    Phase 3 Features:
    - Static obstacle management (barriers, vehicles, pillars)
    - Comprehensive collision detection system
    - Enhanced parking spot validation with tolerances
    - Improved reward engineering with progress tracking
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
        show_sensors: bool = True,        # Whether to visualize sensors
        enable_obstacles: bool = True,    # Whether to include static obstacles
        randomize_target: bool = False,   # Whether to randomize parking target
        randomize_obstacles: bool = False # Whether to randomize obstacle positions
    ):
        super(ParkingEnv, self).__init__()
        
        # Environment parameters
        self.width = width
        self.height = height
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.show_sensors = show_sensors
        self.enable_obstacles = enable_obstacles
        self.randomize_target = randomize_target
        self.randomize_obstacles = randomize_obstacles
        
        # Phase 3: Initialize enhanced modular components
        self.action_space_manager = ActionSpace()
        self.sensor_array = SensorArray(max_range=sensor_max_range)
        self.reward_function = RewardFunction()
        
        # Phase 3: New components
        self.obstacle_manager = ObstacleManager(width, height)
        self.parking_manager = ParkingSpotManager(width, height)
        self.collision_detector = CollisionDetector(self.obstacle_manager, width, height)
        
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
        
        # Phase 3: Enhanced parking spot management
        self.default_target_x = width * 0.8  # Near right side
        self.default_target_y = height * 0.5  # Middle
        self.default_target_theta = 0.0      # Facing right
        
        # Create default parking spot
        self.parking_manager.create_default_spot(
            self.default_target_x, 
            self.default_target_y, 
            self.default_target_theta
        )
        
        # Episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_count = 0
        
        # Phase 3: Enhanced collision tracking
        self.collision_occurred = False
        self.success_achieved = False
        
        # Renderer
        self.renderer = None
        if render_mode:
            self.renderer = ParkingRenderer(width, height)
            
        # Initialize obstacles if enabled
        if self.enable_obstacles:
            self._setup_obstacles()
            
    def _setup_obstacles(self):
        """Setup static obstacles in the environment."""
        active_spot = self.parking_manager.get_active_spot()
        if active_spot:
            # Create obstacles while avoiding the parking area
            obstacles_created = self.obstacle_manager.create_default_obstacles(
                active_spot.x, active_spot.y
            )
            print(f"Created {obstacles_created} static obstacles")
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state with Phase 3 enhancements."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.episode_count += 1
        self.collision_occurred = False
        self.success_achieved = False
        
        # Phase 3: Optionally randomize parking target
        if self.randomize_target:
            self._randomize_parking_target()
        
        # Phase 3: Optionally randomize obstacles
        if self.randomize_obstacles and self.enable_obstacles:
            self._randomize_obstacles()
        
        # Reset car to valid initial position (avoiding obstacles)
        self._reset_car_position()
        
        # Reset subsystems
        self.reward_function._reset_progress_tracking()
        self.collision_detector.clear_history()
        
        return self._get_observation()
    
    def _randomize_parking_target(self):
        """Randomize the parking target location."""
        # Clear existing spots
        self.parking_manager.clear_spots()
        
        # Create new random target (avoiding edges)
        margin = 5.0
        target_x = self.np_random.uniform(margin, self.width - margin)
        target_y = self.np_random.uniform(margin, self.height - margin)
        target_theta = self.np_random.uniform(-np.pi, np.pi)
        
        self.parking_manager.create_default_spot(target_x, target_y, target_theta)
        
    def _randomize_obstacles(self):
        """Randomize obstacle positions."""
        active_spot = self.parking_manager.get_active_spot()
        if active_spot:
            self.obstacle_manager.clear_obstacles()
            self.obstacle_manager.create_default_obstacles(active_spot.x, active_spot.y)
    
    def _reset_car_position(self):
        """Reset car to a valid initial position avoiding obstacles."""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            # Random position not too close to target
            x = self.np_random.uniform(0.1 * self.width, 0.6 * self.width)
            y = self.np_random.uniform(0.1 * self.height, 0.9 * self.height)
            theta = self.np_random.uniform(-np.pi, np.pi)
            
            # Check distance from target
            active_spot = self.parking_manager.get_active_spot()
            if active_spot:
                distance = math.sqrt((x - active_spot.x)**2 + (y - active_spot.y)**2)
                if distance < 5.0:  # Too close to target
                    attempts += 1
                    continue
                    
            # Phase 3: Check for collision with obstacles
            if self.enable_obstacles:
                if not self.collision_detector.is_position_valid(x, y, theta):
                    attempts += 1
                    continue
                    
            # Valid position found
            self.car.reset(x, y, theta, 0.0)
            return
            
        # Fallback to default position if no valid position found
        print("Warning: Could not find valid starting position, using default")
        self.car.reset(self.width * 0.1, self.height * 0.5, 0.0, 0.0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step with Phase 3 enhancements."""
        
        # Validate action
        if not self.action_space_manager.is_valid_action(action):
            raise ValueError(f"Invalid action: {action}. Must be 0-{self.action_space_manager.n_actions-1}")
        
        # Apply action to car using action space system
        self.action_space_manager.apply_action_to_car(self.car, action, self.dt)
        
        # Update car physics
        self.car.update(self.dt)
        
        # Phase 3: Enhanced collision detection
        collision_info = self.collision_detector.check_collision(
            self.car.x, self.car.y, self.car.theta, self.current_step * self.dt
        )
        is_collision = collision_info.collision_type.value != "no_collision"
        
        if is_collision:
            self.collision_occurred = True
        
        # Get sensor readings with obstacle integration
        sensor_readings = self._get_distance_readings()
        
        # Check boundary conditions
        is_out_of_bounds = self._is_out_of_bounds()
        
        # Phase 3: Check parking success using parking manager
        is_successful = self.parking_manager.check_parking_success(
            self.car.x, self.car.y, self.car.theta
        )
        
        if is_successful:
            self.success_achieved = True
        
        # Get parking progress for enhanced reward calculation
        progress_info = self.parking_manager.get_parking_progress(
            self.car.x, self.car.y, self.car.theta
        )
        
        # Calculate reward using enhanced reward system
        active_spot = self.parking_manager.get_active_spot()
        reward_info = self.reward_function.calculate_reward(
            car_x=self.car.x,
            car_y=self.car.y,
            car_theta=self.car.theta,
            car_velocity=self.car.velocity,
            target_x=active_spot.x if active_spot else self.default_target_x,
            target_y=active_spot.y if active_spot else self.default_target_y,
            target_theta=active_spot.angle if active_spot else self.default_target_theta,
            is_collision=is_collision,
            is_out_of_bounds=is_out_of_bounds,
            sensor_readings=sensor_readings,
            timestep=self.current_step,
            episode_done=False
        )
        
        reward = reward_info['total_reward']
        self.total_reward += reward
        
        # Check if episode is done
        done = self._is_done(is_collision, is_out_of_bounds, is_successful)
        
        # Update reward function if episode is done
        if done:
            self.reward_function.calculate_reward(
                car_x=self.car.x, car_y=self.car.y, car_theta=self.car.theta,
                car_velocity=self.car.velocity,
                target_x=active_spot.x if active_spot else self.default_target_x,
                target_y=active_spot.y if active_spot else self.default_target_y,
                target_theta=active_spot.angle if active_spot else self.default_target_theta,
                is_collision=is_collision, is_out_of_bounds=is_out_of_bounds,
                sensor_readings=sensor_readings, timestep=self.current_step,
                episode_done=True
            )
        
        # Get observation
        obs = self._get_observation()
        
        # Phase 3: Enhanced info dictionary
        info = {
            'step': self.current_step,
            'episode': self.episode_count,
            'total_reward': self.total_reward,
            'reward_components': reward_info['components'],
            'distance_to_target': reward_info['distance_to_target'],
            'angle_error': reward_info['angle_error'],
            'angle_error_degrees': reward_info.get('angle_error_degrees', 0),
            'is_collision': is_collision,
            'is_out_of_bounds': is_out_of_bounds,
            'is_successful': is_successful,
            'collision_type': collision_info.collision_type.value,
            'action_description': self.action_space_manager.get_action_description(action),
            'sensor_readings': sensor_readings,
            'min_sensor_distance': min(sensor_readings),
            'collision_risk': self.sensor_array.detect_collision_risk(
                threshold_distance=2.0, 
                car_velocity=self.car.velocity
            ),
            'progress_score': progress_info['progress_score'],
            'parking_accuracy': self.parking_manager.get_detailed_accuracy(
                self.car.x, self.car.y, self.car.theta
            ) if active_spot else {},
            'obstacles_count': len(self.obstacle_manager.obstacles),
            'closest_obstacle_distance': self.collision_detector.get_closest_obstacle_distance(
                self.car.x, self.car.y, self.car.theta
            ) if self.enable_obstacles else float('inf')
        }
        
        self.current_step += 1
        
        return obs, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment with Phase 3 enhancements."""
        if self.renderer is None:
            self.renderer = ParkingRenderer(self.width, self.height)
        
        # Get sensor readings for visualization
        sensor_readings = self._get_distance_readings()
        
        # Render with enhanced components
        return self.renderer.render(
            car=self.car,
            target_x=self.parking_manager.get_active_spot().x if self.parking_manager.get_active_spot() else self.default_target_x,
            target_y=self.parking_manager.get_active_spot().y if self.parking_manager.get_active_spot() else self.default_target_y,
            target_theta=self.parking_manager.get_active_spot().angle if self.parking_manager.get_active_spot() else self.default_target_theta,
            sensor_readings=sensor_readings if self.show_sensors else None,
            mode=mode,
            # Phase 3: Additional rendering data
            obstacles=self.obstacle_manager.obstacles if self.enable_obstacles else [],
            parking_spots=self.parking_manager.parking_spots,
            collision_info=getattr(self, '_last_collision_info', None)
        )
    
    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation with Phase 3 sensor integration."""
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
        """Get 8-directional distance sensor readings with Phase 3 obstacle integration."""
        if self.enable_obstacles:
            # Use obstacle manager for ray intersection
            return self.sensor_array.get_distance_readings_with_obstacles(
                self.car.x, self.car.y, self.car.theta,
                self.obstacle_manager
            )
        else:
            # Use boundary-only readings
            return self.sensor_array.get_distance_readings(
                self.car.x, self.car.y, self.car.theta,
                environment_bounds=(0, 0, self.width, self.height)
            )
    
    def _is_collision(self) -> bool:
        """Check for collision using Phase 3 collision detection system."""
        if self.enable_obstacles:
            collision_info = self.collision_detector.check_collision(
                self.car.x, self.car.y, self.car.theta
            )
            return collision_info.collision_type.value != "no_collision"
        else:
            # Legacy boundary collision check
            return self._is_out_of_bounds()
    
    def _is_out_of_bounds(self) -> bool:
        """Check if car is outside environment boundaries."""
        car_bbox = self.collision_detector.car_model.get_bounding_box(
            self.car.x, self.car.y, self.car.theta
        )
        min_x, min_y, max_x, max_y = car_bbox
        
        return (min_x < 0 or max_x > self.width or 
                min_y < 0 or max_y > self.height)
    
    def _is_done(self, is_collision: bool, is_out_of_bounds: bool, is_successful: bool) -> bool:
        """Check if episode should terminate."""
        # Episode termination conditions from paper:
        # 1. Collision (penalty: -100)
        # 2. Success (reward: +100) 
        # 3. Out of bounds
        # 4. Maximum steps reached
        # 5. Timeout without progress
        
        if is_collision or is_successful or is_out_of_bounds:
            return True
            
        if self.current_step >= self.max_steps:
            return True
            
        return False

    # Phase 3: Enhanced analysis and debugging methods
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        active_spot = self.parking_manager.get_active_spot()
        
        return {
            'environment_size': (self.width, self.height),
            'obstacles_enabled': self.enable_obstacles,
            'obstacles_info': self.obstacle_manager.get_obstacles_summary() if self.enable_obstacles else {},
            'parking_spots_info': self.parking_manager.get_spots_summary(),
            'active_target': {
                'x': active_spot.x if active_spot else self.default_target_x,
                'y': active_spot.y if active_spot else self.default_target_y,
                'theta': active_spot.angle if active_spot else self.default_target_theta,
                'position_tolerance': active_spot.position_tolerance if active_spot else 0.5,
                'angle_tolerance_deg': active_spot.angle_tolerance if active_spot else 10.0
            },
            'collision_detection_enabled': self.enable_obstacles,
            'collision_stats': self.collision_detector.get_collision_statistics() if self.enable_obstacles else {},
            'episode_count': self.episode_count,
            'randomization': {
                'target': self.randomize_target,
                'obstacles': self.randomize_obstacles
            }
        }
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information for analysis."""
        return self.action_space_manager.get_action_space_info()
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get sensor system information."""
        current_readings = self._get_distance_readings()
        return {
            'sensor_count': self.sensor_array.n_sensors,
            'max_range': self.sensor_array.max_range,
            'sensor_angles': [math.degrees(angle) for angle in self.sensor_array.sensor_angles],
            'current_readings': current_readings,
            'min_reading': min(current_readings),
            'max_reading': max(current_readings),
            'collision_risk_detected': self.sensor_array.detect_collision_risk(2.0, self.car.velocity)
        }
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get reward system information."""
        stats = self.reward_function.get_reward_statistics()
        return {
            'reward_parameters': {
                'collision_penalty': self.reward_function.collision_penalty,
                'success_reward': self.reward_function.success_reward,
                'progress_positive': self.reward_function.progress_reward_positive,
                'progress_negative': self.reward_function.progress_reward_negative,
                'time_penalty': self.reward_function.time_penalty
            },
            'current_episode_total': self.total_reward,
            'statistics': stats
        }
    
    def test_all_actions(self) -> Dict[int, str]:
        """Test all actions and return their descriptions."""
        return {i: self.action_space_manager.get_action_description(i) 
                for i in range(self.action_space_manager.n_actions)}
    
    def get_current_state_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of current state for debugging."""
        active_spot = self.parking_manager.get_active_spot()
        sensor_readings = self._get_distance_readings()
        
        analysis = {
            'car_state': {
                'position': (self.car.x, self.car.y),
                'orientation_rad': self.car.theta,
                'orientation_deg': math.degrees(self.car.theta),
                'velocity': self.car.velocity,
                'steering_angle': getattr(self.car, 'steering_angle', 0.0)
            },
            'target_state': {
                'position': (active_spot.x if active_spot else self.default_target_x,
                           active_spot.y if active_spot else self.default_target_y),
                'orientation_rad': active_spot.angle if active_spot else self.default_target_theta,
                'orientation_deg': math.degrees(active_spot.angle) if active_spot else math.degrees(self.default_target_theta)
            },
            'distances_and_errors': {
                'distance_to_target': math.sqrt((self.car.x - (active_spot.x if active_spot else self.default_target_x))**2 + 
                                              (self.car.y - (active_spot.y if active_spot else self.default_target_y))**2),
                'angle_error_rad': self._angle_difference(self.car.theta, active_spot.angle if active_spot else self.default_target_theta),
                'angle_error_deg': math.degrees(self._angle_difference(self.car.theta, active_spot.angle if active_spot else self.default_target_theta))
            },
            'sensors': {
                'readings': sensor_readings,
                'min_distance': min(sensor_readings),
                'sensor_angles_deg': [math.degrees(angle) for angle in self.sensor_array.sensor_angles]
            },
            'status_checks': {
                'is_collision': self._is_collision(),
                'is_out_of_bounds': self._is_out_of_bounds(),
                'is_successful': self.parking_manager.check_parking_success(self.car.x, self.car.y, self.car.theta) if active_spot else False,
                'collision_risk': self.sensor_array.detect_collision_risk(2.0, self.car.velocity)
            },
            'episode_info': {
                'current_step': self.current_step,
                'episode_count': self.episode_count,
                'total_reward': self.total_reward,
                'collision_occurred': self.collision_occurred,
                'success_achieved': self.success_achieved
            }
        }
        
        # Add Phase 3 specific information
        if self.enable_obstacles:
            analysis['obstacles'] = {
                'count': len(self.obstacle_manager.obstacles),
                'closest_distance': self.collision_detector.get_closest_obstacle_distance(
                    self.car.x, self.car.y, self.car.theta
                ),
                'collision_detected': self._is_collision()
            }
        
        if active_spot:
            analysis['parking_accuracy'] = self.parking_manager.get_detailed_accuracy(
                self.car.x, self.car.y, self.car.theta
            )
            analysis['progress_info'] = self.parking_manager.get_parking_progress(
                self.car.x, self.car.y, self.car.theta
            )
        
        return analysis
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """Calculate the difference between two angles, handling wrap-around."""
        diff = angle1 - angle2
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff
    
    def reset_environment_config(self, **kwargs):
        """Reset environment configuration parameters."""
        if 'enable_obstacles' in kwargs:
            self.enable_obstacles = kwargs['enable_obstacles']
            if self.enable_obstacles:
                self._setup_obstacles()
            else:
                self.obstacle_manager.clear_obstacles()
                
        if 'randomize_target' in kwargs:
            self.randomize_target = kwargs['randomize_target']
            
        if 'randomize_obstacles' in kwargs:
            self.randomize_obstacles = kwargs['randomize_obstacles']
            
        if 'show_sensors' in kwargs:
            self.show_sensors = kwargs['show_sensors']
    
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