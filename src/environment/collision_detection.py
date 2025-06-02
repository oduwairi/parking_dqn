"""
Collision Detection System
Handles collision detection between car agent and static obstacles for Phase 3.

Based on Phase 3 objectives:
- Collision penalty: -100 (episode termination)
- Accurate collision detection for various obstacle shapes
- Integration with car physics and obstacle management
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import from other modules
from .obstacles import Obstacle, ObstacleType, ObstacleManager


class CollisionType(Enum):
    """Types of collisions that can occur."""
    NO_COLLISION = "no_collision"
    OBSTACLE_COLLISION = "obstacle_collision"
    BOUNDARY_COLLISION = "boundary_collision"
    SENSOR_DETECTION = "sensor_detection"


@dataclass
class CollisionInfo:
    """
    Information about a collision event.
    """
    collision_type: CollisionType
    collision_point: Tuple[float, float]  # Point of collision
    obstacle: Optional[Obstacle] = None   # Obstacle involved (if any)
    penetration_depth: float = 0.0        # How deep the collision is
    collision_normal: Tuple[float, float] = (0.0, 0.0)  # Collision normal vector
    timestamp: float = 0.0                # When collision occurred


class CarCollisionModel:
    """
    Collision model for the car agent.
    
    Represents the car as a rectangular shape for collision detection.
    """
    
    def __init__(self, length: float = 4.0, width: float = 2.0):
        """
        Initialize car collision model.
        
        Args:
            length: Car length in meters
            width: Car width in meters
        """
        self.length = length
        self.width = width
        
        # Safety margins for collision detection
        self.collision_margin = 0.1  # Extra margin for safety
        
        # Effective dimensions including margin
        self.effective_length = length + 2 * self.collision_margin
        self.effective_width = width + 2 * self.collision_margin
    
    def get_corners(self, x: float, y: float, angle: float) -> List[Tuple[float, float]]:
        """
        Get car corner positions for collision detection.
        
        Args:
            x, y: Car center position
            angle: Car orientation in radians
            
        Returns:
            List of (x, y) corner positions
        """
        half_length = self.effective_length / 2
        half_width = self.effective_width / 2
        
        # Local corner positions (car coordinate frame)
        local_corners = [
            (-half_length, -half_width),  # Rear-left
            (+half_length, -half_width),  # Front-left
            (+half_length, +half_width),  # Front-right
            (-half_length, +half_width),  # Rear-right
        ]
        
        # Apply rotation and translation
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        global_corners = []
        for lx, ly in local_corners:
            # Rotate and translate to global coordinates
            gx = x + (lx * cos_a - ly * sin_a)
            gy = y + (lx * sin_a + ly * cos_a)
            global_corners.append((gx, gy))
        
        return global_corners
    
    def get_bounding_box(self, x: float, y: float, angle: float) -> Tuple[float, float, float, float]:
        """
        Get axis-aligned bounding box for quick collision checking.
        
        Args:
            x, y: Car center position
            angle: Car orientation in radians
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        corners = self.get_corners(x, y, angle)
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_collision_circle(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        Get approximate collision circle for fast collision checking.
        
        Args:
            x, y: Car center position
            
        Returns:
            Tuple of (center_x, center_y, radius)
        """
        # Use diagonal as diameter for conservative collision circle
        radius = math.sqrt(self.effective_length**2 + self.effective_width**2) / 2
        return (x, y, radius)


class CollisionDetector:
    """
    Main collision detection system.
    
    Handles collision detection between car and obstacles, boundaries, etc.
    """
    
    def __init__(self, obstacle_manager: ObstacleManager, environment_width: float, environment_height: float):
        """
        Initialize collision detector.
        
        Args:
            obstacle_manager: Manager for static obstacles
            environment_width: Environment width in meters
            environment_height: Environment height in meters
        """
        self.obstacle_manager = obstacle_manager
        self.environment_width = environment_width
        self.environment_height = environment_height
        
        # Car collision model
        self.car_model = CarCollisionModel()
        
        # Collision detection settings
        self.enable_boundary_collision = True
        self.enable_obstacle_collision = True
        self.collision_tolerance = 0.01  # Minimum distance for collision
        
        # Collision history for analysis
        self.collision_history: List[CollisionInfo] = []
        self.last_collision_time = 0.0
    
    def check_collision(self, car_x: float, car_y: float, car_angle: float, 
                       timestamp: float = 0.0) -> CollisionInfo:
        """
        Check for collisions at the given car position.
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            timestamp: Current simulation time
            
        Returns:
            CollisionInfo with collision details
        """
        # Check boundary collisions first (faster)
        boundary_collision = self._check_boundary_collision(car_x, car_y, car_angle)
        if boundary_collision.collision_type != CollisionType.NO_COLLISION:
            boundary_collision.timestamp = timestamp
            self._record_collision(boundary_collision)
            return boundary_collision
        
        # Check obstacle collisions
        if self.enable_obstacle_collision:
            obstacle_collision = self._check_obstacle_collision(car_x, car_y, car_angle)
            if obstacle_collision.collision_type != CollisionType.NO_COLLISION:
                obstacle_collision.timestamp = timestamp
                self._record_collision(obstacle_collision)
                return obstacle_collision
        
        # No collision detected
        return CollisionInfo(
            collision_type=CollisionType.NO_COLLISION,
            collision_point=(car_x, car_y),
            timestamp=timestamp
        )
    
    def is_position_valid(self, car_x: float, car_y: float, car_angle: float) -> bool:
        """
        Check if a car position is valid (no collisions).
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            
        Returns:
            True if position is valid (no collisions)
        """
        collision_info = self.check_collision(car_x, car_y, car_angle)
        return collision_info.collision_type == CollisionType.NO_COLLISION
    
    def get_closest_obstacle_distance(self, car_x: float, car_y: float, car_angle: float) -> float:
        """
        Get distance to the closest obstacle from car position.
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            
        Returns:
            Distance to closest obstacle in meters
        """
        min_distance = float('inf')
        
        # Get car corners for detailed distance calculation
        car_corners = self.car_model.get_corners(car_x, car_y, car_angle)
        
        # Check distance to each obstacle
        for obstacle in self.obstacle_manager.obstacles:
            obstacle_distance = self._distance_to_obstacle(car_corners, obstacle)
            if obstacle_distance < min_distance:
                min_distance = obstacle_distance
        
        # Check distance to boundaries
        boundary_distance = self._distance_to_boundaries(car_corners)
        if boundary_distance < min_distance:
            min_distance = boundary_distance
        
        return max(0.0, min_distance)  # Ensure non-negative
    
    def _check_boundary_collision(self, car_x: float, car_y: float, car_angle: float) -> CollisionInfo:
        """Check collision with environment boundaries."""
        car_bbox = self.car_model.get_bounding_box(car_x, car_y, car_angle)
        min_x, min_y, max_x, max_y = car_bbox
        
        # Check each boundary
        collision_point = (car_x, car_y)
        
        # Left boundary
        if min_x < 0:
            return CollisionInfo(
                collision_type=CollisionType.BOUNDARY_COLLISION,
                collision_point=(0, car_y),
                penetration_depth=abs(min_x),
                collision_normal=(1.0, 0.0)  # Normal pointing inward
            )
        
        # Right boundary
        if max_x > self.environment_width:
            return CollisionInfo(
                collision_type=CollisionType.BOUNDARY_COLLISION,
                collision_point=(self.environment_width, car_y),
                penetration_depth=max_x - self.environment_width,
                collision_normal=(-1.0, 0.0)  # Normal pointing inward
            )
        
        # Bottom boundary
        if min_y < 0:
            return CollisionInfo(
                collision_type=CollisionType.BOUNDARY_COLLISION,
                collision_point=(car_x, 0),
                penetration_depth=abs(min_y),
                collision_normal=(0.0, 1.0)  # Normal pointing inward
            )
        
        # Top boundary
        if max_y > self.environment_height:
            return CollisionInfo(
                collision_type=CollisionType.BOUNDARY_COLLISION,
                collision_point=(car_x, self.environment_height),
                penetration_depth=max_y - self.environment_height,
                collision_normal=(0.0, -1.0)  # Normal pointing inward
            )
        
        # No boundary collision
        return CollisionInfo(
            collision_type=CollisionType.NO_COLLISION,
            collision_point=collision_point
        )
    
    def _check_obstacle_collision(self, car_x: float, car_y: float, car_angle: float) -> CollisionInfo:
        """Check collision with static obstacles."""
        car_corners = self.car_model.get_corners(car_x, car_y, car_angle)
        
        # Check collision with each obstacle
        for obstacle in self.obstacle_manager.obstacles:
            if self._car_obstacle_collision(car_corners, obstacle):
                # Calculate collision details
                collision_point = self._get_collision_point(car_corners, obstacle)
                penetration_depth = self._get_penetration_depth(car_corners, obstacle)
                collision_normal = self._get_collision_normal(car_corners, obstacle)
                
                return CollisionInfo(
                    collision_type=CollisionType.OBSTACLE_COLLISION,
                    collision_point=collision_point,
                    obstacle=obstacle,
                    penetration_depth=penetration_depth,
                    collision_normal=collision_normal
                )
        
        # No obstacle collision
        return CollisionInfo(
            collision_type=CollisionType.NO_COLLISION,
            collision_point=(car_x, car_y)
        )
    
    def _car_obstacle_collision(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> bool:
        """Check if car corners collide with obstacle."""
        if obstacle.obstacle_type == ObstacleType.CIRCULAR:
            return self._car_circle_collision(car_corners, obstacle)
        else:
            return self._car_rectangle_collision(car_corners, obstacle)
    
    def _car_circle_collision(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> bool:
        """Check collision between car (rectangle) and circular obstacle."""
        # Check if any car corner is inside the circle
        for corner_x, corner_y in car_corners:
            distance = math.sqrt((corner_x - obstacle.x)**2 + (corner_y - obstacle.y)**2)
            if distance <= obstacle.radius:
                return True
        
        # Check if circle center is inside car rectangle
        if self._point_in_polygon(obstacle.x, obstacle.y, car_corners):
            return True
        
        # Check if circle intersects car edges
        for i in range(len(car_corners)):
            p1 = car_corners[i]
            p2 = car_corners[(i + 1) % len(car_corners)]
            
            if self._circle_line_intersection(obstacle.x, obstacle.y, obstacle.radius, p1, p2):
                return True
        
        return False
    
    def _car_rectangle_collision(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> bool:
        """Check collision between car rectangle and rectangular obstacle."""
        obstacle_corners = obstacle.get_corners()
        
        # Check if any car corner is inside obstacle
        for corner_x, corner_y in car_corners:
            if self._point_in_polygon(corner_x, corner_y, obstacle_corners):
                return True
        
        # Check if any obstacle corner is inside car
        for corner_x, corner_y in obstacle_corners:
            if self._point_in_polygon(corner_x, corner_y, car_corners):
                return True
        
        # Check edge intersections (SAT - Separating Axis Theorem)
        # For simplicity, we'll use the above checks which cover most cases
        return False
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _circle_line_intersection(self, cx: float, cy: float, radius: float, 
                                 p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if circle intersects with line segment."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from p1 to circle center
        fx = cx - x1
        fy = cy - y1
        
        # Project circle center onto line
        if dx != 0 or dy != 0:
            t = (fx * dx + fy * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))  # Clamp to line segment
            
            # Closest point on line segment
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            
            # Distance from circle center to closest point
            distance = math.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
            return distance <= radius
        
        return False
    
    def _get_collision_point(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> Tuple[float, float]:
        """Get the collision point between car and obstacle."""
        # Simplified: return obstacle center
        return (obstacle.x, obstacle.y)
    
    def _get_penetration_depth(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> float:
        """Get penetration depth of collision."""
        # Simplified: return small value for now
        return 0.1
    
    def _get_collision_normal(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> Tuple[float, float]:
        """Get collision normal vector."""
        # Simplified: return unit vector from obstacle to car center
        car_center_x = sum(x for x, y in car_corners) / len(car_corners)
        car_center_y = sum(y for x, y in car_corners) / len(car_corners)
        
        normal_x = car_center_x - obstacle.x
        normal_y = car_center_y - obstacle.y
        length = math.sqrt(normal_x**2 + normal_y**2)
        
        if length > 0:
            return (normal_x / length, normal_y / length)
        else:
            return (1.0, 0.0)
    
    def _distance_to_obstacle(self, car_corners: List[Tuple[float, float]], obstacle: Obstacle) -> float:
        """Calculate minimum distance from car to obstacle."""
        min_distance = float('inf')
        
        # Check distance from each car corner to obstacle
        for corner_x, corner_y in car_corners:
            if obstacle.obstacle_type == ObstacleType.CIRCULAR:
                distance = math.sqrt((corner_x - obstacle.x)**2 + (corner_y - obstacle.y)**2) - obstacle.radius
            else:
                # Distance to rectangular obstacle (simplified)
                min_x, min_y, max_x, max_y = obstacle.get_bounding_box()
                closest_x = max(min_x, min(corner_x, max_x))
                closest_y = max(min_y, min(corner_y, max_y))
                distance = math.sqrt((corner_x - closest_x)**2 + (corner_y - closest_y)**2)
            
            if distance < min_distance:
                min_distance = distance
        
        return max(0.0, min_distance)
    
    def _distance_to_boundaries(self, car_corners: List[Tuple[float, float]]) -> float:
        """Calculate minimum distance from car to environment boundaries."""
        min_distance = float('inf')
        
        for corner_x, corner_y in car_corners:
            # Distance to each boundary
            distances = [
                corner_x,  # Left boundary
                self.environment_width - corner_x,  # Right boundary
                corner_y,  # Bottom boundary
                self.environment_height - corner_y  # Top boundary
            ]
            
            corner_min_distance = min(distances)
            if corner_min_distance < min_distance:
                min_distance = corner_min_distance
        
        return max(0.0, min_distance)
    
    def _record_collision(self, collision_info: CollisionInfo):
        """Record collision in history for analysis."""
        self.collision_history.append(collision_info)
        self.last_collision_time = collision_info.timestamp
        
        # Keep history size manageable
        if len(self.collision_history) > 1000:
            self.collision_history = self.collision_history[-500:]
    
    def get_collision_statistics(self) -> Dict[str, Any]:
        """Get collision statistics for analysis."""
        if not self.collision_history:
            return {
                'total_collisions': 0,
                'collision_rate': 0.0,
                'by_type': {},
                'average_penetration': 0.0
            }
        
        total_collisions = len(self.collision_history)
        by_type = {}
        total_penetration = 0.0
        
        for collision in self.collision_history:
            collision_type = collision.collision_type.value
            if collision_type not in by_type:
                by_type[collision_type] = 0
            by_type[collision_type] += 1
            total_penetration += collision.penetration_depth
        
        return {
            'total_collisions': total_collisions,
            'collision_rate': total_collisions / max(1, self.last_collision_time) if self.last_collision_time > 0 else 0,
            'by_type': by_type,
            'average_penetration': total_penetration / total_collisions if total_collisions > 0 else 0.0,
            'last_collision_time': self.last_collision_time
        }
    
    def clear_history(self):
        """Clear collision history."""
        self.collision_history.clear()
        self.last_collision_time = 0.0
    
    def __str__(self) -> str:
        """String representation of collision detector."""
        stats = self.get_collision_statistics()
        return f"CollisionDetector(collisions={stats['total_collisions']}, rate={stats['collision_rate']:.3f})"
    
    def __repr__(self) -> str:
        """Representation of collision detector."""
        return f"CollisionDetector(obstacles={len(self.obstacle_manager.obstacles)}, env={self.environment_width}x{self.environment_height})"


# Global instance for easy access
default_collision_detector = None 