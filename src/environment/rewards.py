"""
Reward Function Implementation
Implements the comprehensive reward function for autonomous parking training.

Based on the reward function from the research paper:
"Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"

Reward Components:
- Collision Penalty: -100 (episode termination)
- Success Reward: +100 (episode termination)  
- Progress Reward: +1 (closer to target), -0.5 (further)
- Time Penalty: -0.1 (per timestep)

Tolerances:
- Position tolerance ε_p = 0.5m
- Orientation tolerance ε_θ = 10°
"""

import math
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from enum import Enum


class RewardType(Enum):
    """Types of rewards that can be given."""
    COLLISION = "collision"
    SUCCESS = "success"
    PROGRESS = "progress"
    TIME = "time"
    BOUNDARY = "boundary"
    VELOCITY = "velocity"


class RewardFunction:
    """
    Comprehensive reward function for autonomous parking.
    
    Implements the reward structure described in the research paper
    with additional components for training stability.
    """
    
    def __init__(
        self,
        collision_penalty: float = -100.0,
        success_reward: float = +100.0,
        progress_reward_positive: float = +1.0,
        progress_reward_negative: float = -0.5,
        time_penalty: float = -0.1,
        position_tolerance: float = 0.5,  # ε_p = 0.5m
        orientation_tolerance_deg: float = 10.0,  # ε_θ = 10°
        boundary_penalty: float = -10.0,
        velocity_reward_scale: float = 0.1
    ):
        """
        Initialize reward function with paper-specified parameters.
        
        Args:
            collision_penalty: Penalty for collision (-100)
            success_reward: Reward for successful parking (+100)
            progress_reward_positive: Reward for moving closer (+1)
            progress_reward_negative: Penalty for moving away (-0.5)
            time_penalty: Penalty per timestep (-0.1)
            position_tolerance: Position tolerance in meters (0.5)
            orientation_tolerance_deg: Orientation tolerance in degrees (10)
            boundary_penalty: Penalty for going out of bounds
            velocity_reward_scale: Scale factor for velocity-based rewards
        """
        # Core reward parameters from paper
        self.collision_penalty = collision_penalty
        self.success_reward = success_reward
        self.progress_reward_positive = progress_reward_positive
        self.progress_reward_negative = progress_reward_negative
        self.time_penalty = time_penalty
        
        # Tolerances from paper
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = math.radians(orientation_tolerance_deg)
        
        # Additional reward components
        self.boundary_penalty = boundary_penalty
        self.velocity_reward_scale = velocity_reward_scale
        
        # Tracking for progress calculation
        self.last_distance_to_target: Optional[float] = None
        self.last_angle_to_target: Optional[float] = None
        
        # Reward history for analysis
        self.reward_history: List[Dict[str, float]] = []
        
    def calculate_reward(
        self,
        car_x: float,
        car_y: float,
        car_theta: float,
        car_velocity: float,
        target_x: float,
        target_y: float,
        target_theta: float,
        is_collision: bool,
        is_out_of_bounds: bool,
        sensor_readings: List[float],
        timestep: int,
        episode_done: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive reward for current state.
        
        Args:
            car_x, car_y: Car position
            car_theta: Car orientation (radians)
            car_velocity: Car velocity
            target_x, target_y: Target parking position
            target_theta: Target parking orientation (radians)
            is_collision: Whether collision occurred
            is_out_of_bounds: Whether car is outside environment
            sensor_readings: List of 8 distance sensor readings
            timestep: Current timestep in episode
            episode_done: Whether episode is terminated
            
        Returns:
            Dictionary containing total reward and component breakdown
        """
        reward_components = {}
        total_reward = 0.0
        
        # 1. Collision Penalty (highest priority, episode termination)
        if is_collision:
            reward_components[RewardType.COLLISION.value] = self.collision_penalty
            total_reward += self.collision_penalty
            
            # Reset progress tracking on collision
            self.last_distance_to_target = None
            self.last_angle_to_target = None
            
        # 2. Success Reward (highest positive, episode termination)
        elif self._is_successful_parking(car_x, car_y, car_theta, target_x, target_y, target_theta):
            reward_components[RewardType.SUCCESS.value] = self.success_reward
            total_reward += self.success_reward
            
        # 3. Boundary Penalty
        elif is_out_of_bounds:
            reward_components[RewardType.BOUNDARY.value] = self.boundary_penalty
            total_reward += self.boundary_penalty
            
        else:
            # 4. Progress Reward (only if no collision/success/boundary violation)
            progress_reward = self._calculate_progress_reward(
                car_x, car_y, car_theta, target_x, target_y, target_theta
            )
            if progress_reward != 0:
                reward_components[RewardType.PROGRESS.value] = progress_reward
                total_reward += progress_reward
            
            # 5. Time Penalty (encourage efficiency)
            reward_components[RewardType.TIME.value] = self.time_penalty
            total_reward += self.time_penalty
            
            # 6. Velocity-based reward (encourage smooth motion)
            velocity_reward = self._calculate_velocity_reward(car_velocity, sensor_readings)
            if velocity_reward != 0:
                reward_components[RewardType.VELOCITY.value] = velocity_reward
                total_reward += velocity_reward
        
        # Create reward info dictionary
        reward_info = {
            'total_reward': total_reward,
            'components': reward_components,
            'distance_to_target': self._distance_to_target(car_x, car_y, target_x, target_y),
            'angle_error': self._angle_error(car_theta, target_theta),
            'is_successful': self._is_successful_parking(car_x, car_y, car_theta, target_x, target_y, target_theta),
            'timestep': timestep
        }
        
        # Store in history for analysis
        self.reward_history.append(reward_info.copy())
        
        return reward_info
    
    def _is_successful_parking(
        self, 
        car_x: float, car_y: float, car_theta: float,
        target_x: float, target_y: float, target_theta: float
    ) -> bool:
        """
        Check if parking is successful based on position and orientation tolerances.
        
        Args:
            car_x, car_y: Car position
            car_theta: Car orientation
            target_x, target_y: Target position
            target_theta: Target orientation
            
        Returns:
            True if parking is successful within tolerances
        """
        # Position tolerance check: ||(x,y) - (x_target, y_target)|| ≤ ε_p
        position_error = self._distance_to_target(car_x, car_y, target_x, target_y)
        position_ok = position_error <= self.position_tolerance
        
        # Orientation tolerance check: |θ - θ_target| ≤ ε_θ
        angle_error = abs(self._angle_error(car_theta, target_theta))
        orientation_ok = angle_error <= self.orientation_tolerance
        
        return position_ok and orientation_ok
    
    def _distance_to_target(self, car_x: float, car_y: float, target_x: float, target_y: float) -> float:
        """Calculate Euclidean distance to target position."""
        return math.sqrt((car_x - target_x)**2 + (car_y - target_y)**2)
    
    def _angle_error(self, car_theta: float, target_theta: float) -> float:
        """Calculate angular error with proper wrapping."""
        error = car_theta - target_theta
        
        # Normalize to [-π, π]
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
            
        return error
    
    def _calculate_progress_reward(
        self,
        car_x: float, car_y: float, car_theta: float,
        target_x: float, target_y: float, target_theta: float
    ) -> float:
        """
        Calculate progress reward based on movement toward/away from target.
        
        Returns:
            +1.0 if closer to target, -0.5 if further, 0 if same
        """
        current_distance = self._distance_to_target(car_x, car_y, target_x, target_y)
        current_angle_error = abs(self._angle_error(car_theta, target_theta))
        
        # Combine position and orientation for overall progress metric
        # Weight position more heavily than orientation
        current_progress = current_distance + 0.3 * current_angle_error
        
        if self.last_distance_to_target is not None:
            last_progress = self.last_distance_to_target + 0.3 * (self.last_angle_to_target or 0)
            
            progress_delta = last_progress - current_progress
            
            if progress_delta > 0.1:  # Meaningful improvement
                reward = self.progress_reward_positive
            elif progress_delta < -0.1:  # Meaningful degradation
                reward = self.progress_reward_negative
            else:  # No significant change
                reward = 0.0
        else:
            reward = 0.0  # No previous state to compare
        
        # Update tracking
        self.last_distance_to_target = current_distance
        self.last_angle_to_target = current_angle_error
        
        return reward
    
    def _calculate_velocity_reward(self, velocity: float, sensor_readings: List[float]) -> float:
        """
        Calculate velocity-based reward to encourage smooth motion.
        
        Args:
            velocity: Current car velocity
            sensor_readings: Distance sensor readings
            
        Returns:
            Small reward/penalty for velocity behavior
        """
        # Get minimum sensor distance (closest obstacle)
        min_distance = min(sensor_readings) if sensor_readings else 20.0
        
        # Encourage slower speeds near obstacles
        if min_distance < 3.0:  # Close to obstacle
            if abs(velocity) < 1.0:  # Moving slowly - good
                return 0.05 * self.velocity_reward_scale
            else:  # Moving fast near obstacle - bad
                return -0.1 * self.velocity_reward_scale
        
        # Encourage moderate speeds in open space
        elif min_distance > 10.0:  # Open space
            if 1.0 <= abs(velocity) <= 3.0:  # Good speed
                return 0.02 * self.velocity_reward_scale
            elif abs(velocity) < 0.1:  # Too slow in open space
                return -0.02 * self.velocity_reward_scale
        
        return 0.0  # Neutral for other cases
    
    def reset_progress_tracking(self):
        """Reset progress tracking for new episode."""
        self.last_distance_to_target = None
        self.last_angle_to_target = None
    
    def get_reward_statistics(self, last_n_episodes: int = 100) -> Dict[str, Any]:
        """
        Get statistics about reward distribution.
        
        Args:
            last_n_episodes: Number of recent episodes to analyze
            
        Returns:
            Dictionary with reward statistics
        """
        if not self.reward_history:
            return {'error': 'No reward history available'}
        
        recent_rewards = self.reward_history[-last_n_episodes:] if last_n_episodes > 0 else self.reward_history
        
        # Extract total rewards
        total_rewards = [entry['total_reward'] for entry in recent_rewards]
        
        # Count different reward types
        component_counts = {}
        component_totals = {}
        
        for entry in recent_rewards:
            for reward_type, value in entry['components'].items():
                if reward_type not in component_counts:
                    component_counts[reward_type] = 0
                    component_totals[reward_type] = 0.0
                component_counts[reward_type] += 1
                component_totals[reward_type] += value
        
        statistics = {
            'total_episodes_analyzed': len(recent_rewards),
            'average_total_reward': np.mean(total_rewards) if total_rewards else 0.0,
            'std_total_reward': np.std(total_rewards) if total_rewards else 0.0,
            'min_total_reward': np.min(total_rewards) if total_rewards else 0.0,
            'max_total_reward': np.max(total_rewards) if total_rewards else 0.0,
            'component_counts': component_counts,
            'component_averages': {k: v / component_counts[k] for k, v in component_totals.items()},
            'success_rate': component_counts.get('success', 0) / len(recent_rewards) if recent_rewards else 0.0,
            'collision_rate': component_counts.get('collision', 0) / len(recent_rewards) if recent_rewards else 0.0,
        }
        
        return statistics
    
    def clear_history(self):
        """Clear reward history to save memory."""
        self.reward_history.clear()
    
    def __str__(self) -> str:
        """String representation of reward function."""
        lines = ["Reward Function Configuration:"]
        lines.append(f"  Collision Penalty: {self.collision_penalty}")
        lines.append(f"  Success Reward: {self.success_reward}")
        lines.append(f"  Progress Reward: +{self.progress_reward_positive} / {self.progress_reward_negative}")
        lines.append(f"  Time Penalty: {self.time_penalty}")
        lines.append(f"  Position Tolerance: {self.position_tolerance}m")
        lines.append(f"  Orientation Tolerance: {math.degrees(self.orientation_tolerance):.1f}°")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representation of reward function."""
        return (f"RewardFunction(collision={self.collision_penalty}, "
                f"success={self.success_reward}, tolerance_pos={self.position_tolerance}m)")


# Global instance for easy access
default_reward_function = RewardFunction() 