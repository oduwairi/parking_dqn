"""
Static Obstacle Management
Handles static obstacles (barriers, stationary vehicles) for Phase 3.

Based on Phase 3 objectives from the roadmap:
- Add static obstacles (barriers, stationary vehicles)
- Implement comprehensive collision detection
- Create diverse training scenarios

Obstacle Types:
- Rectangular barriers (walls, fences)
- Circular obstacles (pillars, posts)
- Vehicle obstacles (parked cars)
"""

import math
import numpy as np
import pygame
from typing import List, Tuple, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass


class ObstacleType(Enum):
    """Types of static obstacles."""
    BARRIER = "barrier"           # Rectangular barriers/walls
    CIRCULAR = "circular"         # Circular obstacles/pillars
    VEHICLE = "vehicle"          # Parked vehicles
    BOUNDARY = "boundary"        # Environment boundaries


@dataclass
class Obstacle:
    """
    Base obstacle class with common properties.
    """
    x: float                     # Center X position
    y: float                     # Center Y position
    obstacle_type: ObstacleType  # Type of obstacle
    width: float = 0.0          # Width (for rectangular obstacles)
    height: float = 0.0         # Height (for rectangular obstacles)
    radius: float = 0.0         # Radius (for circular obstacles)
    angle: float = 0.0          # Rotation angle in radians
    color: Tuple[int, int, int] = (100, 100, 100)  # RGB color for rendering
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get axis-aligned bounding box for quick collision checking.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if self.obstacle_type == ObstacleType.CIRCULAR:
            return (
                self.x - self.radius,
                self.y - self.radius,
                self.x + self.radius,
                self.y + self.radius
            )
        else:  # Rectangular obstacles
            # For simplicity, assume axis-aligned bounding box even for rotated rectangles
            half_width = self.width / 2
            half_height = self.height / 2
            return (
                self.x - half_width,
                self.y - half_height,
                self.x + half_width,
                self.y + half_height
            )
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Get corner positions for rectangular obstacles.
        
        Returns:
            List of (x, y) corner positions
        """
        if self.obstacle_type == ObstacleType.CIRCULAR:
            # Return bounding box corners for circular obstacles
            min_x, min_y, max_x, max_y = self.get_bounding_box()
            return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        
        # Local corner positions (obstacle coordinate frame)
        half_width = self.width / 2
        half_height = self.height / 2
        local_corners = [
            (-half_width, -half_height),  # Bottom-left
            (+half_width, -half_height),  # Bottom-right
            (+half_width, +half_height),  # Top-right
            (-half_width, +half_height),  # Top-left
        ]
        
        # Apply rotation if needed
        if abs(self.angle) > 0.001:
            cos_a = math.cos(self.angle)
            sin_a = math.sin(self.angle)
            rotated_corners = []
            for lx, ly in local_corners:
                rx = lx * cos_a - ly * sin_a
                ry = lx * sin_a + ly * cos_a
                rotated_corners.append((rx, ry))
            local_corners = rotated_corners
        
        # Transform to global coordinates
        global_corners = []
        for lx, ly in local_corners:
            global_corners.append((self.x + lx, self.y + ly))
        
        return global_corners


class ObstacleManager:
    """
    Manages all static obstacles in the environment.
    
    Provides obstacle placement, collision detection, and rendering.
    """
    
    def __init__(self, environment_width: float, environment_height: float):
        """
        Initialize obstacle manager.
        
        Args:
            environment_width: Environment width in meters
            environment_height: Environment height in meters
        """
        self.environment_width = environment_width
        self.environment_height = environment_height
        self.obstacles: List[Obstacle] = []
        
        # Obstacle creation parameters
        self.min_obstacle_distance = 3.0  # Minimum distance between obstacles
        self.parking_clearance = 8.0      # Clearance around parking spots
        
    def clear_obstacles(self):
        """Remove all obstacles."""
        self.obstacles.clear()
    
    def add_obstacle(self, obstacle: Obstacle) -> bool:
        """
        Add an obstacle to the environment.
        
        Args:
            obstacle: Obstacle to add
            
        Returns:
            True if obstacle was added successfully, False if invalid placement
        """
        # Validate obstacle is within environment bounds
        if not self._is_obstacle_in_bounds(obstacle):
            return False
        
        # Check for overlap with existing obstacles
        if self._check_obstacle_overlap(obstacle):
            return False
        
        self.obstacles.append(obstacle)
        return True
    
    def create_default_obstacles(self, parking_x: float, parking_y: float) -> int:
        """
        Create a default set of static obstacles for training.
        
        Args:
            parking_x, parking_y: Target parking spot position
            
        Returns:
            Number of obstacles created
        """
        self.clear_obstacles()
        obstacles_created = 0
        
        # Create boundary walls (environment edges)
        wall_thickness = 0.5
        
        # Top wall
        top_wall = Obstacle(
            x=self.environment_width / 2,
            y=self.environment_height - wall_thickness / 2,
            obstacle_type=ObstacleType.BARRIER,
            width=self.environment_width,
            height=wall_thickness,
            color=(80, 80, 80)
        )
        if self.add_obstacle(top_wall):
            obstacles_created += 1
        
        # Bottom wall
        bottom_wall = Obstacle(
            x=self.environment_width / 2,
            y=wall_thickness / 2,
            obstacle_type=ObstacleType.BARRIER,
            width=self.environment_width,
            height=wall_thickness,
            color=(80, 80, 80)
        )
        if self.add_obstacle(bottom_wall):
            obstacles_created += 1
        
        # Left wall
        left_wall = Obstacle(
            x=wall_thickness / 2,
            y=self.environment_height / 2,
            obstacle_type=ObstacleType.BARRIER,
            width=wall_thickness,
            height=self.environment_height,
            color=(80, 80, 80)
        )
        if self.add_obstacle(left_wall):
            obstacles_created += 1
        
        # Right wall
        right_wall = Obstacle(
            x=self.environment_width - wall_thickness / 2,
            y=self.environment_height / 2,
            obstacle_type=ObstacleType.BARRIER,
            width=wall_thickness,
            height=self.environment_height,
            color=(80, 80, 80)
        )
        if self.add_obstacle(right_wall):
            obstacles_created += 1
        
        # Add some interior obstacles (avoiding parking area)
        interior_obstacles = [
            # Parked vehicles
            Obstacle(
                x=self.environment_width * 0.3,
                y=self.environment_height * 0.2,
                obstacle_type=ObstacleType.VEHICLE,
                width=4.0, height=2.0,
                color=(150, 50, 50)  # Red car
            ),
            Obstacle(
                x=self.environment_width * 0.3,
                y=self.environment_height * 0.8,
                obstacle_type=ObstacleType.VEHICLE,
                width=4.0, height=2.0,
                color=(50, 150, 50)  # Green car
            ),
            # Circular pillars
            Obstacle(
                x=self.environment_width * 0.5,
                y=self.environment_height * 0.3,
                obstacle_type=ObstacleType.CIRCULAR,
                radius=1.0,
                color=(100, 100, 150)  # Blue pillar
            ),
            Obstacle(
                x=self.environment_width * 0.5,
                y=self.environment_height * 0.7,
                obstacle_type=ObstacleType.CIRCULAR,
                radius=1.0,
                color=(100, 100, 150)  # Blue pillar
            ),
            # Rectangular barriers
            Obstacle(
                x=self.environment_width * 0.15,
                y=self.environment_height * 0.5,
                obstacle_type=ObstacleType.BARRIER,
                width=1.0, height=6.0,
                color=(120, 120, 120)  # Gray barrier
            ),
        ]
        
        # Add interior obstacles that don't conflict with parking
        for obs in interior_obstacles:
            # Check distance from parking spot
            distance_to_parking = math.sqrt(
                (obs.x - parking_x)**2 + (obs.y - parking_y)**2
            )
            
            if distance_to_parking > self.parking_clearance:
                if self.add_obstacle(obs):
                    obstacles_created += 1
        
        return obstacles_created
    
    def check_point_collision(self, x: float, y: float) -> Optional[Obstacle]:
        """
        Check if a point collides with any obstacle.
        
        Args:
            x, y: Point coordinates
            
        Returns:
            Obstacle that collides with point, or None if no collision
        """
        for obstacle in self.obstacles:
            if self._point_in_obstacle(x, y, obstacle):
                return obstacle
        return None
    
    def check_circle_collision(self, x: float, y: float, radius: float) -> List[Obstacle]:
        """
        Check if a circle collides with any obstacles.
        
        Args:
            x, y: Circle center
            radius: Circle radius
            
        Returns:
            List of obstacles that collide with the circle
        """
        colliding_obstacles = []
        
        for obstacle in self.obstacles:
            if self._circle_obstacle_collision(x, y, radius, obstacle):
                colliding_obstacles.append(obstacle)
        
        return colliding_obstacles
    
    def check_rectangle_collision(self, corners: List[Tuple[float, float]]) -> List[Obstacle]:
        """
        Check if a rectangle (defined by corners) collides with any obstacles.
        
        Args:
            corners: List of (x, y) corner positions
            
        Returns:
            List of obstacles that collide with the rectangle
        """
        colliding_obstacles = []
        
        for obstacle in self.obstacles:
            if self._rectangle_obstacle_collision(corners, obstacle):
                colliding_obstacles.append(obstacle)
        
        return colliding_obstacles
    
    def get_ray_intersection(self, start_x: float, start_y: float, angle: float, max_distance: float) -> float:
        """
        Cast a ray and find the distance to the nearest obstacle intersection.
        
        Args:
            start_x, start_y: Ray starting position
            angle: Ray direction in radians
            max_distance: Maximum ray distance
            
        Returns:
            Distance to nearest intersection, or max_distance if no intersection
        """
        min_distance = max_distance
        
        # Ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        for obstacle in self.obstacles:
            distance = self._ray_obstacle_intersection(start_x, start_y, dx, dy, obstacle, max_distance)
            if distance < min_distance:
                min_distance = distance
        
        return min_distance
    
    def _is_obstacle_in_bounds(self, obstacle: Obstacle) -> bool:
        """Check if obstacle is within environment bounds."""
        min_x, min_y, max_x, max_y = obstacle.get_bounding_box()
        return (min_x >= 0 and min_y >= 0 and 
                max_x <= self.environment_width and max_y <= self.environment_height)
    
    def _check_obstacle_overlap(self, new_obstacle: Obstacle) -> bool:
        """Check if new obstacle overlaps with existing obstacles."""
        for existing in self.obstacles:
            if self._obstacles_overlap(new_obstacle, existing):
                return True
        return False
    
    def _obstacles_overlap(self, obs1: Obstacle, obs2: Obstacle) -> bool:
        """Check if two obstacles overlap."""
        # Simple bounding box overlap check
        box1 = obs1.get_bounding_box()
        box2 = obs2.get_bounding_box()
        
        return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                   box1[3] < box2[1] or box1[1] > box2[3])
    
    def _point_in_obstacle(self, x: float, y: float, obstacle: Obstacle) -> bool:
        """Check if point is inside obstacle."""
        if obstacle.obstacle_type == ObstacleType.CIRCULAR:
            distance = math.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2)
            return distance <= obstacle.radius
        else:
            # Point-in-polygon test for rectangular obstacles
            corners = obstacle.get_corners()
            return self._point_in_polygon(x, y, corners)
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
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
    
    def _circle_obstacle_collision(self, cx: float, cy: float, radius: float, obstacle: Obstacle) -> bool:
        """Check collision between circle and obstacle."""
        if obstacle.obstacle_type == ObstacleType.CIRCULAR:
            # Circle-circle collision
            distance = math.sqrt((cx - obstacle.x)**2 + (cy - obstacle.y)**2)
            return distance <= (radius + obstacle.radius)
        else:
            # Circle-rectangle collision (simplified)
            min_x, min_y, max_x, max_y = obstacle.get_bounding_box()
            
            # Find closest point on rectangle to circle center
            closest_x = max(min_x, min(cx, max_x))
            closest_y = max(min_y, min(cy, max_y))
            
            # Check distance to closest point
            distance = math.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
            return distance <= radius
    
    def _rectangle_obstacle_collision(self, corners: List[Tuple[float, float]], obstacle: Obstacle) -> bool:
        """Check collision between rectangle and obstacle."""
        # Check if any corner of rectangle is inside obstacle
        for corner_x, corner_y in corners:
            if self._point_in_obstacle(corner_x, corner_y, obstacle):
                return True
        
        # Check if any corner of obstacle is inside rectangle  
        obstacle_corners = obstacle.get_corners()
        for obs_x, obs_y in obstacle_corners:
            if self._point_in_polygon(obs_x, obs_y, corners):
                return True
        
        return False
    
    def _ray_obstacle_intersection(self, start_x: float, start_y: float, dx: float, dy: float, 
                                  obstacle: Obstacle, max_distance: float) -> float:
        """Calculate ray-obstacle intersection distance."""
        if obstacle.obstacle_type == ObstacleType.CIRCULAR:
            return self._ray_circle_intersection(start_x, start_y, dx, dy, obstacle, max_distance)
        else:
            return self._ray_rectangle_intersection(start_x, start_y, dx, dy, obstacle, max_distance)
    
    def _ray_circle_intersection(self, start_x: float, start_y: float, dx: float, dy: float,
                                obstacle: Obstacle, max_distance: float) -> float:
        """Calculate ray-circle intersection distance."""
        # Vector from ray start to circle center
        to_center_x = obstacle.x - start_x
        to_center_y = obstacle.y - start_y
        
        # Project onto ray direction
        proj_length = to_center_x * dx + to_center_y * dy
        
        if proj_length < 0:  # Circle is behind ray start
            return max_distance
        
        # Find closest point on ray to circle center
        closest_x = start_x + proj_length * dx
        closest_y = start_y + proj_length * dy
        
        # Distance from circle center to closest point on ray
        distance_to_ray = math.sqrt((obstacle.x - closest_x)**2 + (obstacle.y - closest_y)**2)
        
        if distance_to_ray > obstacle.radius:  # Ray misses circle
            return max_distance
        
        # Calculate intersection distance
        half_chord = math.sqrt(obstacle.radius**2 - distance_to_ray**2)
        intersection_distance = proj_length - half_chord
        
        if intersection_distance < 0:  # Ray starts inside circle
            intersection_distance = 0
        
        return min(intersection_distance, max_distance)
    
    def _ray_rectangle_intersection(self, start_x: float, start_y: float, dx: float, dy: float,
                                   obstacle: Obstacle, max_distance: float) -> float:
        """Calculate ray-rectangle intersection distance."""
        min_x, min_y, max_x, max_y = obstacle.get_bounding_box()
        
        # Calculate intersection with each edge
        distances = []
        
        # Left edge
        if dx != 0:
            t = (min_x - start_x) / dx
            if t > 0:
                y_intersect = start_y + t * dy
                if min_y <= y_intersect <= max_y:
                    distances.append(t)
        
        # Right edge
        if dx != 0:
            t = (max_x - start_x) / dx
            if t > 0:
                y_intersect = start_y + t * dy
                if min_y <= y_intersect <= max_y:
                    distances.append(t)
        
        # Bottom edge
        if dy != 0:
            t = (min_y - start_y) / dy
            if t > 0:
                x_intersect = start_x + t * dx
                if min_x <= x_intersect <= max_x:
                    distances.append(t)
        
        # Top edge
        if dy != 0:
            t = (max_y - start_y) / dy
            if t > 0:
                x_intersect = start_x + t * dx
                if min_x <= x_intersect <= max_x:
                    distances.append(t)
        
        if distances:
            return min(min(distances), max_distance)
        else:
            return max_distance
    
    def render(self, surface: pygame.Surface, scale: float):
        """
        Render all obstacles on pygame surface.
        
        Args:
            surface: Pygame surface to draw on
            scale: Scale factor for coordinate conversion
        """
        for obstacle in self.obstacles:
            self._render_obstacle(surface, obstacle, scale)
    
    def _render_obstacle(self, surface: pygame.Surface, obstacle: Obstacle, scale: float):
        """Render a single obstacle."""
        if obstacle.obstacle_type == ObstacleType.CIRCULAR:
            # Render circular obstacle
            center_x = int(obstacle.x * scale)
            center_y = int(obstacle.y * scale)
            radius = int(obstacle.radius * scale)
            pygame.draw.circle(surface, obstacle.color, (center_x, center_y), radius)
            pygame.draw.circle(surface, (0, 0, 0), (center_x, center_y), radius, 2)  # Border
        else:
            # Render rectangular obstacle
            corners = obstacle.get_corners()
            screen_corners = [(int(x * scale), int(y * scale)) for x, y in corners]
            pygame.draw.polygon(surface, obstacle.color, screen_corners)
            pygame.draw.polygon(surface, (0, 0, 0), screen_corners, 2)  # Border
    
    def get_obstacles_summary(self) -> Dict[str, Any]:
        """Get summary information about all obstacles."""
        summary = {
            'total_obstacles': len(self.obstacles),
            'by_type': {},
            'total_area_covered': 0.0,
            'obstacles': []
        }
        
        for obstacle in self.obstacles:
            obs_type = obstacle.obstacle_type.value
            if obs_type not in summary['by_type']:
                summary['by_type'][obs_type] = 0
            summary['by_type'][obs_type] += 1
            
            # Calculate area
            if obstacle.obstacle_type == ObstacleType.CIRCULAR:
                area = math.pi * obstacle.radius**2
            else:
                area = obstacle.width * obstacle.height
            summary['total_area_covered'] += area
            
            # Add obstacle info
            summary['obstacles'].append({
                'type': obs_type,
                'position': (obstacle.x, obstacle.y),
                'size': (obstacle.width, obstacle.height) if obstacle.obstacle_type != ObstacleType.CIRCULAR else obstacle.radius,
                'area': area
            })
        
        return summary
    
    def __str__(self) -> str:
        """String representation of obstacle manager."""
        summary = self.get_obstacles_summary()
        lines = [f"ObstacleManager: {summary['total_obstacles']} obstacles"]
        for obs_type, count in summary['by_type'].items():
            lines.append(f"  {obs_type}: {count}")
        lines.append(f"  Total area: {summary['total_area_covered']:.1f} m²")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representation of obstacle manager."""
        return f"ObstacleManager(obstacles={len(self.obstacles)}, env={self.environment_width}x{self.environment_height})"


# Global instance for easy access
default_obstacle_manager = None 