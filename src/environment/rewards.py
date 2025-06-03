"""
Reward Function Implementation - Enhanced for Phase 3
Implements the comprehensive reward function for autonomous parking training.

Based on the reward function from the research paper:
"Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"

Phase 3 Enhancements:
- Enhanced progress tracking toward parking goal
- Integration with obstacle collision detection
- Better shaping for goal-directed behavior
- Parking accuracy metrics

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
    PROXIMITY = "proximity"
    ORIENTATION = "orientation"


class RewardFunction:
    """
    Enhanced comprehensive reward function for autonomous parking (Phase 3).
    
    Implements the reward structure described in the research paper
    with enhanced progress tracking and goal-directed behavior.
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
        velocity_reward_scale: float = 0.1,
        proximity_reward_scale: float = 0.5,
        orientation_reward_scale: float = 0.3
    ):
        """
        Initialize enhanced reward function with paper-specified parameters.
        
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
            proximity_reward_scale: Scale for proximity-based shaping
            orientation_reward_scale: Scale for orientation-based shaping
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
        self.proximity_reward_scale = proximity_reward_scale
        self.orientation_reward_scale = orientation_reward_scale
        
        # Enhanced progress tracking for Phase 3
        self.last_distance_to_target: Optional[float] = None
        self.last_angle_to_target: Optional[float] = None
        self.last_combined_progress: Optional[float] = None
        self.best_distance_achieved: Optional[float] = None
        self.best_angle_achieved: Optional[float] = None
        
        # Reward history for analysis
        self.reward_history: List[Dict[str, float]] = []
        self.episode_rewards: List[float] = []
        
        # Progress tracking parameters
        self.progress_threshold = 0.05  # Minimum change to be considered progress
        self.stagnation_penalty = -0.2  # Penalty for no progress over time
        self.stagnation_steps = 50     # Steps before applying stagnation penalty
        self.steps_without_progress = 0
        
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
        Calculate TRULY SPARSE reward - only genuine progress matters!
        
        Philosophy: No free rewards! Agent must EARN every positive point.
        
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
        
        # Calculate current state metrics
        distance_to_target = self._distance_to_target(car_x, car_y, target_x, target_y)
        angle_error = abs(self._angle_error(car_theta, target_theta))
        
        # 1. COLLISION = INSTANT DEATH (-100)
        if is_collision:
            reward_components[RewardType.COLLISION.value] = self.collision_penalty
            total_reward += self.collision_penalty
            self._reset_progress_tracking()
            
        # 2. SUCCESS = BIG WIN (+100)
        elif self._is_successful_parking(car_x, car_y, car_theta, target_x, target_y, target_theta):
            reward_components[RewardType.SUCCESS.value] = self.success_reward
            total_reward += self.success_reward
            
        # 3. OUT OF BOUNDS = PENALTY (-10)
        elif is_out_of_bounds:
            reward_components[RewardType.BOUNDARY.value] = self.boundary_penalty
            total_reward += self.boundary_penalty
            
        else:
            # 4. CORE PHILOSOPHY: Only reward SIGNIFICANT, GENUINE progress
            
            # TIME PENALTY: Every step costs something (encourages efficiency)
            time_penalty = -0.2  # Doubled from -0.1
            reward_components[RewardType.TIME.value] = time_penalty
            total_reward += time_penalty
            
            # SPARSE PROGRESS REWARD: Only for substantial improvement
            if self.last_distance_to_target is not None:
                distance_improvement = self.last_distance_to_target - distance_to_target
                
                # Only reward SIGNIFICANT distance improvements (≥0.5m closer)
                if distance_improvement >= 0.5:
                    progress_reward = distance_improvement * 2.0  # Scale with improvement
                    reward_components[RewardType.PROGRESS.value] = progress_reward
                    total_reward += progress_reward
                    self.steps_without_progress = 0
                    
                # Penalize moving away from target (≥0.3m further)
                elif distance_improvement <= -0.3:
                    progress_penalty = distance_improvement * 1.0  # Negative value
                    reward_components[RewardType.PROGRESS.value] = progress_penalty
                    total_reward += progress_penalty
                    self.steps_without_progress += 1
                    
                else:
                    # No reward for tiny movements - neutral
                    self.steps_without_progress += 1
            
            # PROXIMITY BONUS: Only when very close to target
            if distance_to_target < 3.0:  # Only within 3m
                proximity_bonus = (3.0 - distance_to_target) / 3.0 * 0.5  # Max +0.5
                reward_components['proximity_bonus'] = proximity_bonus
                total_reward += proximity_bonus
                
            # ORIENTATION BONUS: Only when close AND well-oriented  
            if distance_to_target < 2.0 and angle_error < math.radians(30):  # Within 2m and 30°
                orientation_bonus = (math.radians(30) - angle_error) / math.radians(30) * 0.3  # Max +0.3
                reward_components['orientation_bonus'] = orientation_bonus
                total_reward += orientation_bonus
            
            # STAGNATION PENALTY: Heavily penalize doing nothing
            if self.steps_without_progress >= 30:  # Reduced from 50
                stagnation_penalty = -0.5 * (self.steps_without_progress - 30) / 20  # Escalating penalty
                stagnation_penalty = max(stagnation_penalty, -3.0)  # Cap at -3.0
                reward_components['stagnation_penalty'] = stagnation_penalty
                total_reward += stagnation_penalty
            
            # SPEED PENALTY: Discourage reckless driving near target
            if distance_to_target < 5.0 and abs(car_velocity) > 2.0:
                speed_penalty = -0.1 * (abs(car_velocity) - 2.0)
                reward_components['speed_penalty'] = speed_penalty
                total_reward += speed_penalty
        
        # Update progress tracking
        self._update_progress_tracking(distance_to_target, angle_error)
        
        # Store reward components for analysis
        if episode_done:
            self.reward_history.append(reward_components)
        
        return {
            'total_reward': total_reward,
            'components': reward_components,
            'distance_to_target': distance_to_target,
            'angle_error': angle_error,
            'success': self._is_successful_parking(car_x, car_y, car_theta, target_x, target_y, target_theta),
            'collision': is_collision,
            'out_of_bounds': is_out_of_bounds,
            'is_terminal': is_collision or episode_done or self._is_successful_parking(car_x, car_y, car_theta, target_x, target_y, target_theta)
        }
    
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
    
    def _calculate_enhanced_progress_reward(
        self,
        car_x: float, car_y: float, car_theta: float,
        target_x: float, target_y: float, target_theta: float
    ) -> float:
        """
        Calculate enhanced progress reward with better goal direction.
        
        Returns:
            +1.0 if closer to target, -0.5 if further, 0 if same
        """
        current_distance = self._distance_to_target(car_x, car_y, target_x, target_y)
        current_angle_error = abs(self._angle_error(car_theta, target_theta))
        
        # Weighted combination: position is more important than orientation
        current_combined_score = current_distance + 0.2 * current_angle_error
        
        if self.last_combined_progress is not None:
            progress_delta = self.last_combined_progress - current_combined_score
            
            if progress_delta > self.progress_threshold:  # Significant improvement
                self.steps_without_progress = 0
                reward = self.progress_reward_positive
                
                # Extra bonus for major improvements
                if progress_delta > 0.5:
                    reward *= 1.5
                    
            elif progress_delta < -self.progress_threshold:  # Significant degradation
                self.steps_without_progress += 1
                reward = self.progress_reward_negative
                
                # Larger penalty for major setbacks
                if progress_delta < -0.5:
                    reward *= 1.5
                    
            else:  # No significant change
                self.steps_without_progress += 1
                reward = 0.0
        else:
            # First step, no reward
            reward = 0.0
            
        return reward
    
    def _calculate_proximity_reward(self, distance_to_target: float) -> float:
        """
        Calculate proximity-based reward shaping (closer = better).
        
        Args:
            distance_to_target: Current distance to parking target
            
        Returns:
            Proximity reward scaled by distance
        """
        # Inverse distance reward with scaling
        # Closer distances get higher rewards, but capped
        max_reward_distance = 20.0  # Maximum distance for reward calculation
        
        if distance_to_target > max_reward_distance:
            return 0.0
        
        # Normalize distance and invert (closer = higher reward)
        normalized_distance = distance_to_target / max_reward_distance
        proximity_score = (1.0 - normalized_distance) * self.proximity_reward_scale
        
        return proximity_score
    
    def _calculate_orientation_reward(self, angle_error: float) -> float:
        """
        Calculate orientation-based reward shaping.
        
        Args:
            angle_error: Current angular error from target orientation
            
        Returns:
            Orientation reward scaled by error
        """
        # Inverse angle error reward
        max_angle_error = math.pi  # 180 degrees
        
        if angle_error > max_angle_error:
            return 0.0
        
        # Normalize angle error and invert
        normalized_angle = angle_error / max_angle_error
        orientation_score = (1.0 - normalized_angle) * self.orientation_reward_scale
        
        return orientation_score
    
    def _calculate_accuracy_bonus(self, distance_error: float, angle_error: float) -> float:
        """
        Calculate bonus reward for high accuracy parking.
        
        Args:
            distance_error: Distance from target position
            angle_error: Angular error from target orientation
            
        Returns:
            Bonus reward for accuracy
        """
        # Bonus for parking within tolerance with high accuracy
        distance_accuracy = max(0.0, (self.position_tolerance - distance_error) / self.position_tolerance)
        angle_accuracy = max(0.0, (self.orientation_tolerance - angle_error) / self.orientation_tolerance)
        
        # Geometric mean of accuracies
        overall_accuracy = math.sqrt(distance_accuracy * angle_accuracy)
        
        # Bonus scales with accuracy (0 to 10 points)
        accuracy_bonus = overall_accuracy * 10.0
        
        return accuracy_bonus
    
    def _calculate_stagnation_penalty(self) -> float:
        """Calculate penalty for staying in same position too long."""
        if self.steps_without_progress >= self.stagnation_steps:
            # Increasing penalty for longer stagnation
            multiplier = (self.steps_without_progress - self.stagnation_steps) / self.stagnation_steps + 1
            return self.stagnation_penalty * min(multiplier, 3.0)  # Cap at 3x penalty
        return 0.0
    
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
    
    def _update_progress_tracking(self, distance_to_target: float, angle_error: float):
        """Update progress tracking variables."""
        # Update last values for next comparison
        self.last_distance_to_target = distance_to_target
        self.last_angle_to_target = angle_error
        self.last_combined_progress = distance_to_target + 0.2 * angle_error
        
        # Track best achievements
        if self.best_distance_achieved is None or distance_to_target < self.best_distance_achieved:
            self.best_distance_achieved = distance_to_target
            
        if self.best_angle_achieved is None or angle_error < self.best_angle_achieved:
            self.best_angle_achieved = angle_error
            
    def _reset_progress_tracking(self):
        """Reset progress tracking variables."""
        self.last_distance_to_target = None
        self.last_angle_to_target = None
        self.last_combined_progress = None
        self.steps_without_progress = 0
    
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