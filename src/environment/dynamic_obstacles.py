"""
Dynamic Obstacles System for Advanced Parking Scenarios
Implements moving obstacles including pedestrians and vehicles for Phase 6 training.

Features:
- Moving pedestrians with realistic behavior patterns
- Moving vehicles with traffic-like movement
- Collision prediction and avoidance
- Configurable obstacle density and behavior
"""

import numpy as np
import pygame
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ObstacleState:
    """State representation for a dynamic obstacle."""
    x: float
    y: float
    vx: float  # Velocity in x direction
    vy: float  # Velocity in y direction
    width: float
    height: float
    obstacle_type: str  # 'pedestrian', 'vehicle', 'bicycle'
    active: bool = True


class DynamicObstacle(ABC):
    """Abstract base class for dynamic obstacles."""
    
    def __init__(self, x: float, y: float, obstacle_type: str):
        self.state = ObstacleState(
            x=x, y=y, vx=0.0, vy=0.0,
            width=1.0, height=1.0,
            obstacle_type=obstacle_type
        )
        self.color = (255, 0, 0)  # Default red
        
    @abstractmethod
    def update(self, dt: float, environment_bounds: Tuple[float, float], car_position: Tuple[float, float]):
        """Update obstacle position and behavior."""
        pass
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get obstacle bounding box (x_min, y_min, x_max, y_max)."""
        half_w = self.state.width / 2
        half_h = self.state.height / 2
        return (
            self.state.x - half_w,
            self.state.y - half_h,
            self.state.x + half_w,
            self.state.y + half_h
        )
    
    def check_collision(self, car_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if obstacle collides with car."""
        if not self.state.active:
            return False
            
        obs_bounds = self.get_bounds()
        car_x1, car_y1, car_x2, car_y2 = car_bounds
        obs_x1, obs_y1, obs_x2, obs_y2 = obs_bounds
        
        return not (car_x2 < obs_x1 or car_x1 > obs_x2 or car_y2 < obs_y1 or car_y1 > obs_y2)


class MovingPedestrian(DynamicObstacle):
    """Moving pedestrian obstacle with realistic walking patterns."""
    
    def __init__(self, x: float, y: float, speed: float = 1.2):
        super().__init__(x, y, "pedestrian")
        self.state.width = 0.6  # 60cm wide
        self.state.height = 0.6  # 60cm deep
        self.speed = speed  # Normal walking speed ~1.2 m/s
        self.color = (255, 165, 0)  # Orange
        
        # Walking behavior
        self.target_x = x + np.random.uniform(-10, 10)
        self.target_y = y + np.random.uniform(-10, 10)
        self.change_direction_timer = 0
        self.direction_change_interval = np.random.uniform(3, 8)  # Change direction every 3-8 seconds
        
    def update(self, dt: float, environment_bounds: Tuple[float, float], car_position: Tuple[float, float]):
        """Update pedestrian movement with realistic walking behavior."""
        if not self.state.active:
            return
            
        env_width, env_height = environment_bounds
        car_x, car_y = car_position
        
        # Change direction periodically
        self.change_direction_timer += dt
        if self.change_direction_timer >= self.direction_change_interval:
            self.target_x = np.random.uniform(2, env_width - 2)
            self.target_y = np.random.uniform(2, env_height - 2)
            self.change_direction_timer = 0
            self.direction_change_interval = np.random.uniform(3, 8)
        
        # Move towards target
        dx = self.target_x - self.state.x
        dy = self.target_y - self.state.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0.5:  # If not at target
            # Normalize direction
            dx /= distance
            dy /= distance
            
            # Car avoidance behavior
            car_distance = np.sqrt((car_x - self.state.x)**2 + (car_y - self.state.y)**2)
            if car_distance < 3.0:  # If car is close, avoid it
                avoid_x = self.state.x - car_x
                avoid_y = self.state.y - car_y
                avoid_distance = np.sqrt(avoid_x*avoid_x + avoid_y*avoid_y)
                if avoid_distance > 0:
                    avoid_x /= avoid_distance
                    avoid_y /= avoid_distance
                    # Blend avoidance with target direction
                    dx = 0.7 * avoid_x + 0.3 * dx
                    dy = 0.7 * avoid_y + 0.3 * dy
            
            # Update velocity
            self.state.vx = dx * self.speed
            self.state.vy = dy * self.speed
        else:
            # Reached target, slow down
            self.state.vx *= 0.8
            self.state.vy *= 0.8
        
        # Update position
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        
        # Keep within bounds
        self.state.x = np.clip(self.state.x, 1, env_width - 1)
        self.state.y = np.clip(self.state.y, 1, env_height - 1)


class MovingVehicle(DynamicObstacle):
    """Moving vehicle obstacle with traffic-like behavior."""
    
    def __init__(self, x: float, y: float, direction: str = "horizontal", speed: float = 3.0):
        super().__init__(x, y, "vehicle")
        self.state.width = 4.5  # Car width
        self.state.height = 2.0  # Car height
        self.speed = speed
        self.direction = direction  # "horizontal", "vertical", "random"
        self.color = (0, 0, 255)  # Blue
        
        # Set initial velocity based on direction
        if direction == "horizontal":
            self.state.vx = speed if np.random.random() > 0.5 else -speed
            self.state.vy = 0
        elif direction == "vertical":
            self.state.vx = 0
            self.state.vy = speed if np.random.random() > 0.5 else -speed
        else:  # random
            angle = np.random.uniform(0, 2 * np.pi)
            self.state.vx = speed * np.cos(angle)
            self.state.vy = speed * np.sin(angle)
    
    def update(self, dt: float, environment_bounds: Tuple[float, float], car_position: Tuple[float, float]):
        """Update vehicle movement with traffic-like behavior."""
        if not self.state.active:
            return
            
        env_width, env_height = environment_bounds
        car_x, car_y = car_position
        
        # Car avoidance - slow down if too close to parking car
        car_distance = np.sqrt((car_x - self.state.x)**2 + (car_y - self.state.y)**2)
        speed_factor = 1.0
        if car_distance < 5.0:
            speed_factor = max(0.3, car_distance / 5.0)  # Slow down when close
        
        # Update position
        self.state.x += self.state.vx * speed_factor * dt
        self.state.y += self.state.vy * speed_factor * dt
        
        # Boundary behavior - wrap around or bounce
        if self.state.x < -self.state.width:
            self.state.x = env_width + self.state.width
        elif self.state.x > env_width + self.state.width:
            self.state.x = -self.state.width
            
        if self.state.y < -self.state.height:
            self.state.y = env_height + self.state.height
        elif self.state.y > env_height + self.state.height:
            self.state.y = -self.state.height


class Cyclist(DynamicObstacle):
    """Cycling obstacle with bicycle-like movement patterns."""
    
    def __init__(self, x: float, y: float, speed: float = 2.5):
        super().__init__(x, y, "bicycle")
        self.state.width = 1.8  # Bicycle length
        self.state.height = 0.6  # Bicycle width
        self.speed = speed
        self.color = (0, 255, 0)  # Green
        
        # Cycling behavior
        self.path_points = self._generate_cycling_path(x, y)
        self.current_target_idx = 0
        
    def _generate_cycling_path(self, start_x: float, start_y: float) -> List[Tuple[float, float]]:
        """Generate a cycling path with gentle curves."""
        path = [(start_x, start_y)]
        
        for i in range(5):
            # Create curved path
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            distance = np.random.uniform(8, 15)
            
            last_x, last_y = path[-1]
            new_x = last_x + distance * np.cos(angle)
            new_y = last_y + distance * np.sin(angle)
            path.append((new_x, new_y))
        
        return path
    
    def update(self, dt: float, environment_bounds: Tuple[float, float], car_position: Tuple[float, float]):
        """Update cyclist movement along curved path."""
        if not self.state.active:
            return
            
        if self.current_target_idx >= len(self.path_points):
            # Generate new path
            self.path_points = self._generate_cycling_path(self.state.x, self.state.y)
            self.current_target_idx = 0
        
        # Move towards current target
        target_x, target_y = self.path_points[self.current_target_idx]
        dx = target_x - self.state.x
        dy = target_y - self.state.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < 1.0:  # Reached target
            self.current_target_idx += 1
        else:
            # Move towards target
            dx /= distance
            dy /= distance
            self.state.vx = dx * self.speed
            self.state.vy = dy * self.speed
            
            # Update position
            self.state.x += self.state.vx * dt
            self.state.y += self.state.vy * dt


class DynamicObstacleManager:
    """Manages all dynamic obstacles in the environment."""
    
    def __init__(self, environment_bounds: Tuple[float, float]):
        self.environment_bounds = environment_bounds
        self.obstacles: List[DynamicObstacle] = []
        self.spawn_timer = 0
        self.spawn_interval = 5.0  # Spawn new obstacle every 5 seconds
        
    def add_obstacle(self, obstacle: DynamicObstacle):
        """Add a dynamic obstacle to the environment."""
        self.obstacles.append(obstacle)
    
    def spawn_random_obstacle(self):
        """Spawn a random obstacle at environment edge."""
        env_width, env_height = self.environment_bounds
        obstacle_type = np.random.choice(['pedestrian', 'vehicle', 'bicycle'], p=[0.5, 0.3, 0.2])
        
        # Spawn at random edge
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        
        if edge == 'top':
            x, y = np.random.uniform(0, env_width), 0
        elif edge == 'bottom':
            x, y = np.random.uniform(0, env_width), env_height
        elif edge == 'left':
            x, y = 0, np.random.uniform(0, env_height)
        else:  # right
            x, y = env_width, np.random.uniform(0, env_height)
        
        if obstacle_type == 'pedestrian':
            obstacle = MovingPedestrian(x, y, speed=np.random.uniform(0.8, 1.5))
        elif obstacle_type == 'vehicle':
            direction = 'horizontal' if edge in ['top', 'bottom'] else 'vertical'
            obstacle = MovingVehicle(x, y, direction=direction, speed=np.random.uniform(2, 4))
        else:  # bicycle
            obstacle = Cyclist(x, y, speed=np.random.uniform(2, 3))
        
        self.add_obstacle(obstacle)
    
    def update(self, dt: float, car_position: Tuple[float, float]):
        """Update all dynamic obstacles."""
        # Update existing obstacles
        for obstacle in self.obstacles[:]:  # Copy list to allow removal
            obstacle.update(dt, self.environment_bounds, car_position)
            
            # Remove obstacles that are far from environment
            if (obstacle.state.x < -10 or obstacle.state.x > self.environment_bounds[0] + 10 or
                obstacle.state.y < -10 or obstacle.state.y > self.environment_bounds[1] + 10):
                self.obstacles.remove(obstacle)
        
        # Spawn new obstacles
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval and len(self.obstacles) < 8:  # Max 8 obstacles
            self.spawn_random_obstacle()
            self.spawn_timer = 0
    
    def check_collisions(self, car_bounds: Tuple[float, float, float, float]) -> bool:
        """Check if car collides with any dynamic obstacle."""
        for obstacle in self.obstacles:
            if obstacle.check_collision(car_bounds):
                return True
        return False
    
    def get_sensor_distances(self, car_x: float, car_y: float, sensor_angles: List[float], max_range: float = 10.0) -> List[float]:
        """Get distances to dynamic obstacles for car sensors."""
        distances = [max_range] * len(sensor_angles)
        
        for i, angle in enumerate(sensor_angles):
            min_distance = max_range
            
            # Check ray intersection with all obstacles
            for obstacle in self.obstacles:
                if not obstacle.state.active:
                    continue
                    
                # Simple ray-box intersection
                obs_bounds = obstacle.get_bounds()
                distance = self._ray_box_intersection(car_x, car_y, angle, obs_bounds, max_range)
                min_distance = min(min_distance, distance)
            
            distances[i] = min_distance
        
        return distances
    
    def _ray_box_intersection(self, ray_x: float, ray_y: float, ray_angle: float, 
                             box_bounds: Tuple[float, float, float, float], max_range: float) -> float:
        """Calculate ray-box intersection distance."""
        box_x1, box_y1, box_x2, box_y2 = box_bounds
        
        # Ray direction
        dx = np.cos(ray_angle)
        dy = np.sin(ray_angle)
        
        # Check intersection with each box edge
        for t in np.linspace(0, max_range, int(max_range * 10)):
            x = ray_x + t * dx
            y = ray_y + t * dy
            
            if box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2:
                return t
        
        return max_range
    
    def render(self, screen, scale: float, offset_x: float, offset_y: float):
        """Render all dynamic obstacles."""
        for obstacle in self.obstacles:
            if not obstacle.state.active:
                continue
                
            # Convert world coordinates to screen coordinates
            screen_x = int((obstacle.state.x + offset_x) * scale)
            screen_y = int((obstacle.state.y + offset_y) * scale)
            width = int(obstacle.state.width * scale)
            height = int(obstacle.state.height * scale)
            
            # Draw obstacle
            pygame.draw.rect(screen, obstacle.color, 
                           (screen_x - width//2, screen_y - height//2, width, height))
            
            # Draw velocity vector for debugging
            if abs(obstacle.state.vx) > 0.1 or abs(obstacle.state.vy) > 0.1:
                end_x = screen_x + int(obstacle.state.vx * scale * 2)
                end_y = screen_y + int(obstacle.state.vy * scale * 2)
                pygame.draw.line(screen, (255, 255, 255), (screen_x, screen_y), (end_x, end_y), 2)


# Configuration presets for different dynamic obstacle scenarios
DYNAMIC_OBSTACLE_PRESETS = {
    "light_traffic": {
        "max_obstacles": 3,
        "spawn_interval": 8.0,
        "pedestrian_probability": 0.7,
        "vehicle_probability": 0.2,
        "bicycle_probability": 0.1
    },
    
    "moderate_traffic": {
        "max_obstacles": 5,
        "spawn_interval": 5.0,
        "pedestrian_probability": 0.5,
        "vehicle_probability": 0.3,
        "bicycle_probability": 0.2
    },
    
    "heavy_traffic": {
        "max_obstacles": 8,
        "spawn_interval": 3.0,
        "pedestrian_probability": 0.4,
        "vehicle_probability": 0.4,
        "bicycle_probability": 0.2
    },
    
    "pedestrian_zone": {
        "max_obstacles": 6,
        "spawn_interval": 4.0,
        "pedestrian_probability": 0.8,
        "vehicle_probability": 0.1,
        "bicycle_probability": 0.1
    }
} 