"""
Car Agent Class
Implements car physics and kinematic motion model.

Kinematic equations from the paper:
x_{t+1} = x_t + v_t * cos(θ_t) * Δt
y_{t+1} = y_t + v_t * sin(θ_t) * Δt  
θ_{t+1} = θ_t + (v_t/L) * tan(δ_t) * Δt

Phase 2 Update: Action handling moved to ActionSpace class.
Car agent now focuses only on physics and state management.
"""

import math
import numpy as np
from typing import Tuple


class CarAgent:
    """
    Car agent with kinematic motion model.
    Implements the simplified 2D car physics as described in the paper.
    
    Phase 2: Action handling delegated to ActionSpace class.
    """
    
    def __init__(
        self, 
        x: float, 
        y: float, 
        theta: float, 
        velocity: float, 
        wheelbase: float = 2.5
    ):
        """
        Initialize car agent.
        
        Args:
            x: Initial x position (meters)
            y: Initial y position (meters) 
            theta: Initial orientation angle (radians)
            velocity: Initial velocity (m/s)
            wheelbase: Wheelbase length L (meters), default 2.5m from paper
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity
        self.wheelbase = wheelbase
        
        # Current steering angle (radians)
        self.steering_angle = 0.0
        
        # Velocity and steering limits
        self.max_velocity = 5.0   # m/s
        self.min_velocity = -5.0  # m/s (reverse)
        self.max_steering = math.radians(30)  # 30 degrees max steering
        
        # Friction/damping
        self.velocity_decay = 0.95  # Velocity decay factor per timestep
        self.steering_decay = 0.8   # Steering returns to center
        
    def reset(self, x: float, y: float, theta: float, velocity: float):
        """Reset car to initial state."""
        self.x = x
        self.y = y 
        self.theta = theta
        self.velocity = velocity
        self.steering_angle = 0.0
        
    def update(self, dt: float):
        """
        Update car physics using kinematic motion model.
        
        Args:
            dt: Time step (seconds)
        """
        # Apply kinematic equations from paper
        # x_{t+1} = x_t + v_t * cos(θ_t) * Δt
        self.x += self.velocity * math.cos(self.theta) * dt
        
        # y_{t+1} = y_t + v_t * sin(θ_t) * Δt  
        self.y += self.velocity * math.sin(self.theta) * dt
        
        # θ_{t+1} = θ_t + (v_t/L) * tan(δ_t) * Δt
        if abs(self.velocity) > 0.001:  # Avoid division by zero
            angular_velocity = (self.velocity / self.wheelbase) * math.tan(self.steering_angle)
            self.theta += angular_velocity * dt
            
        # Normalize theta to [-π, π]
        self.theta = self._normalize_angle(self.theta)
        
        # Apply decay/friction
        self.velocity *= self.velocity_decay
        self.steering_angle *= self.steering_decay
        
        # Stop very small velocities (numerical stability)
        if abs(self.velocity) < 0.01:
            self.velocity = 0.0
            
        # Return steering to center when no input
        if abs(self.steering_angle) < 0.01:
            self.steering_angle = 0.0
            
    def get_state(self) -> Tuple[float, float, float, float]:
        """Get current car state as tuple (x, y, theta, velocity)."""
        return (self.x, self.y, self.theta, self.velocity)
        
    def get_position(self) -> Tuple[float, float]:
        """Get current position as tuple (x, y)."""
        return (self.x, self.y)
        
    def get_orientation(self) -> float:
        """Get current orientation angle in radians."""
        return self.theta
        
    def get_velocity(self) -> float:
        """Get current velocity in m/s."""
        return self.velocity
        
    def get_corners(self) -> np.ndarray:
        """
        Get car corner positions for collision detection and rendering.
        Assumes rectangular car with given dimensions.
        
        Returns:
            np.ndarray: 4x2 array of corner positions [[x1,y1], [x2,y2], ...]
        """
        # Car dimensions (simplified)
        length = 4.0  # meters
        width = 2.0   # meters
        
        # Local corner positions (car coordinate frame)
        local_corners = np.array([
            [-length/2, -width/2],  # Rear left
            [+length/2, -width/2],  # Front left  
            [+length/2, +width/2],  # Front right
            [-length/2, +width/2],  # Rear right
        ])
        
        # Rotation matrix
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        
        # Transform to global coordinates
        global_corners = np.dot(local_corners, rotation_matrix.T)
        global_corners[:, 0] += self.x
        global_corners[:, 1] += self.y
        
        return global_corners
        
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
        
    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def __repr__(self) -> str:
        """String representation of car agent."""
        return (f"CarAgent(pos=({self.x:.2f}, {self.y:.2f}), "
                f"θ={math.degrees(self.theta):.1f}°, v={self.velocity:.2f}m/s)") 