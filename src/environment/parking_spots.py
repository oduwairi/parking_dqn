"""
Parking Spot Validation and Management
Handles parking spot detection, validation, and success conditions for Phase 3.

Based on the research paper specifications:
- Position tolerance: ε_p = 0.5m
- Orientation tolerance: ε_θ = 10°
- Success reward: +100 (episode termination)
- Parking accuracy metrics
"""

import math
import numpy as np
import pygame
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ParkingSpotType(Enum):
    """Types of parking spots."""
    PARALLEL = "parallel"         # Parallel parking
    PERPENDICULAR = "perpendicular"  # Perpendicular parking
    ANGLED = "angled"            # Angled parking (30°, 45°, 60°)


@dataclass
class ParkingSpot:
    """
    Parking spot definition with target position and validation.
    """
    x: float                     # Target center X position
    y: float                     # Target center Y position
    angle: float                 # Target orientation in radians
    width: float                 # Spot width (perpendicular to angle)
    length: float                # Spot length (parallel to angle)
    spot_type: ParkingSpotType   # Type of parking spot
    
    # Tolerance parameters (from research paper)
    position_tolerance: float = 0.5    # ε_p = 0.5m
    angle_tolerance: float = 10.0      # ε_θ = 10° (in degrees)
    
    def __post_init__(self):
        """Convert angle tolerance to radians."""
        self.angle_tolerance_rad = math.radians(self.angle_tolerance)
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Get parking spot corner positions for visualization.
        
        Returns:
            List of (x, y) corner positions
        """
        half_width = self.width / 2
        half_length = self.length / 2
        
        # Local corner positions (spot coordinate frame)
        local_corners = [
            (-half_length, -half_width),  # Back-left
            (+half_length, -half_width),  # Front-left
            (+half_length, +half_width),  # Front-right
            (-half_length, +half_width),  # Back-right
        ]
        
        # Apply rotation
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        global_corners = []
        for lx, ly in local_corners:
            # Rotate and translate to global coordinates
            gx = self.x + (lx * cos_a - ly * sin_a)
            gy = self.y + (lx * sin_a + ly * cos_a)
            global_corners.append((gx, gy))
        
        return global_corners
    
    def is_position_valid(self, car_x: float, car_y: float) -> bool:
        """
        Check if car position is within tolerance of parking spot.
        
        Args:
            car_x, car_y: Car center position
            
        Returns:
            True if position is within tolerance
        """
        distance = math.sqrt((car_x - self.x)**2 + (car_y - self.y)**2)
        return distance <= self.position_tolerance
    
    def is_orientation_valid(self, car_angle: float) -> bool:
        """
        Check if car orientation is within tolerance of parking spot.
        
        Args:
            car_angle: Car orientation in radians
            
        Returns:
            True if orientation is within tolerance
        """
        # Normalize angles to [-π, π]
        target_angle = self._normalize_angle(self.angle)
        current_angle = self._normalize_angle(car_angle)
        
        # Calculate angular difference
        angle_diff = abs(target_angle - current_angle)
        
        # Handle wrap-around (e.g., -179° vs 179°)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        return angle_diff <= self.angle_tolerance_rad
    
    def is_parking_successful(self, car_x: float, car_y: float, car_angle: float) -> bool:
        """
        Check if parking is successful (both position and orientation valid).
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            
        Returns:
            True if parking is successful
        """
        return (self.is_position_valid(car_x, car_y) and 
                self.is_orientation_valid(car_angle))
    
    def get_position_error(self, car_x: float, car_y: float) -> float:
        """
        Get position error distance from target.
        
        Args:
            car_x, car_y: Car center position
            
        Returns:
            Distance error in meters
        """
        return math.sqrt((car_x - self.x)**2 + (car_y - self.y)**2)
    
    def get_orientation_error(self, car_angle: float) -> float:
        """
        Get orientation error from target.
        
        Args:
            car_angle: Car orientation in radians
            
        Returns:
            Angular error in radians
        """
        target_angle = self._normalize_angle(self.angle)
        current_angle = self._normalize_angle(car_angle)
        
        angle_diff = abs(target_angle - current_angle)
        
        # Handle wrap-around
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        return angle_diff
    
    def get_parking_accuracy(self, car_x: float, car_y: float, car_angle: float) -> Dict[str, float]:
        """
        Get detailed parking accuracy metrics.
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            
        Returns:
            Dictionary with accuracy metrics
        """
        position_error = self.get_position_error(car_x, car_y)
        orientation_error = self.get_orientation_error(car_angle)
        
        # Calculate accuracy percentages (100% = perfect, 0% = at tolerance limit)
        position_accuracy = max(0.0, (self.position_tolerance - position_error) / self.position_tolerance)
        orientation_accuracy = max(0.0, (self.angle_tolerance_rad - orientation_error) / self.angle_tolerance_rad)
        
        # Overall accuracy (geometric mean)
        overall_accuracy = math.sqrt(position_accuracy * orientation_accuracy)
        
        return {
            'position_error': position_error,
            'orientation_error': orientation_error,
            'orientation_error_degrees': math.degrees(orientation_error),
            'position_accuracy': position_accuracy,
            'orientation_accuracy': orientation_accuracy,
            'overall_accuracy': overall_accuracy,
            'is_successful': self.is_parking_successful(car_x, car_y, car_angle)
        }
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π] range."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def render(self, surface: pygame.Surface, scale: float, color: Tuple[int, int, int] = (0, 255, 0)):
        """
        Render parking spot on pygame surface.
        
        Args:
            surface: Pygame surface to draw on
            scale: Scale factor for coordinate conversion
            color: RGB color for spot outline
        """
        corners = self.get_corners()
        screen_corners = [(int(x * scale), int(y * scale)) for x, y in corners]
        
        # Draw parking spot outline
        pygame.draw.polygon(surface, color, screen_corners, 3)
        
        # Draw target position marker
        center_x = int(self.x * scale)
        center_y = int(self.y * scale)
        pygame.draw.circle(surface, color, (center_x, center_y), 5)
        
        # Draw orientation indicator (arrow)
        arrow_length = min(self.width, self.length) * 0.4 * scale
        end_x = center_x + arrow_length * math.cos(self.angle)
        end_y = center_y + arrow_length * math.sin(self.angle)
        pygame.draw.line(surface, color, (center_x, center_y), (int(end_x), int(end_y)), 3)
    
    def __str__(self) -> str:
        """String representation of parking spot."""
        return (f"ParkingSpot(pos=({self.x:.1f}, {self.y:.1f}), "
                f"angle={math.degrees(self.angle):.1f}°, "
                f"size={self.length:.1f}x{self.width:.1f})")
    
    def __repr__(self) -> str:
        """Representation of parking spot."""
        return (f"ParkingSpot(x={self.x}, y={self.y}, angle={self.angle}, "
                f"width={self.width}, length={self.length}, type={self.spot_type})")


class ParkingSpotManager:
    """
    Manages parking spots and validation logic.
    
    Provides spot creation, validation, and success detection.
    """
    
    def __init__(self, environment_width: float, environment_height: float):
        """
        Initialize parking spot manager.
        
        Args:
            environment_width: Environment width in meters
            environment_height: Environment height in meters
        """
        self.environment_width = environment_width
        self.environment_height = environment_height
        self.parking_spots: List[ParkingSpot] = []
        self.active_spot_index = 0
        
        # Default car dimensions for spot sizing
        self.default_car_length = 4.0
        self.default_car_width = 2.0
        
        # Spot size multipliers (to provide clearance)
        self.length_multiplier = 1.2
        self.width_multiplier = 1.1
    
    def clear_spots(self):
        """Remove all parking spots."""
        self.parking_spots.clear()
        self.active_spot_index = 0
    
    def add_spot(self, spot: ParkingSpot) -> bool:
        """
        Add a parking spot.
        
        Args:
            spot: Parking spot to add
            
        Returns:
            True if spot was added successfully
        """
        # Validate spot is within environment bounds
        if not self._is_spot_in_bounds(spot):
            return False
        
        self.parking_spots.append(spot)
        return True
    
    def create_default_spot(self, target_x: float = None, target_y: float = None, 
                           target_angle: float = None) -> ParkingSpot:
        """
        Create a default parking spot.
        
        Args:
            target_x, target_y: Target position (defaults to center-right)
            target_angle: Target orientation (defaults to 0°)
            
        Returns:
            Created parking spot
        """
        if target_x is None:
            target_x = self.environment_width * 0.8
        if target_y is None:
            target_y = self.environment_height * 0.5
        if target_angle is None:
            target_angle = 0.0  # Facing right
        
        # Calculate spot dimensions
        spot_length = self.default_car_length * self.length_multiplier
        spot_width = self.default_car_width * self.width_multiplier
        
        spot = ParkingSpot(
            x=target_x,
            y=target_y,
            angle=target_angle,
            width=spot_width,
            length=spot_length,
            spot_type=ParkingSpotType.PERPENDICULAR
        )
        
        self.add_spot(spot)
        return spot
    
    def create_parallel_spot(self, x: float, y: float, angle: float) -> ParkingSpot:
        """
        Create a parallel parking spot.
        
        Args:
            x, y: Spot center position
            angle: Spot orientation in radians
            
        Returns:
            Created parking spot
        """
        # Parallel spots are typically longer and narrower
        spot_length = self.default_car_length * 1.5
        spot_width = self.default_car_width * 1.05
        
        spot = ParkingSpot(
            x=x, y=y, angle=angle,
            width=spot_width,
            length=spot_length,
            spot_type=ParkingSpotType.PARALLEL
        )
        
        self.add_spot(spot)
        return spot
    
    def create_angled_spot(self, x: float, y: float, angle: float) -> ParkingSpot:
        """
        Create an angled parking spot.
        
        Args:
            x, y: Spot center position
            angle: Spot orientation in radians
            
        Returns:
            Created parking spot
        """
        # Angled spots have standard dimensions
        spot_length = self.default_car_length * self.length_multiplier
        spot_width = self.default_car_width * self.width_multiplier
        
        spot = ParkingSpot(
            x=x, y=y, angle=angle,
            width=spot_width,
            length=spot_length,
            spot_type=ParkingSpotType.ANGLED
        )
        
        self.add_spot(spot)
        return spot
    
    def get_active_spot(self) -> Optional[ParkingSpot]:
        """
        Get the currently active parking spot.
        
        Returns:
            Active parking spot, or None if no spots exist
        """
        if 0 <= self.active_spot_index < len(self.parking_spots):
            return self.parking_spots[self.active_spot_index]
        return None
    
    def set_active_spot(self, index: int) -> bool:
        """
        Set the active parking spot by index.
        
        Args:
            index: Spot index
            
        Returns:
            True if index is valid
        """
        if 0 <= index < len(self.parking_spots):
            self.active_spot_index = index
            return True
        return False
    
    def check_parking_success(self, car_x: float, car_y: float, car_angle: float) -> bool:
        """
        Check if parking is successful at the active spot.
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            
        Returns:
            True if parking is successful
        """
        active_spot = self.get_active_spot()
        if active_spot is None:
            return False
        
        return active_spot.is_parking_successful(car_x, car_y, car_angle)
    
    def get_parking_progress(self, car_x: float, car_y: float, car_angle: float) -> Dict[str, float]:
        """
        Get parking progress metrics for the active spot.
        
        Args:
            car_x, car_y: Car center position  
            car_angle: Car orientation in radians
            
        Returns:
            Dictionary with progress metrics
        """
        active_spot = self.get_active_spot()
        if active_spot is None:
            return {
                'distance_to_target': float('inf'),
                'orientation_error': float('inf'),
                'progress_score': 0.0,
                'is_successful': False
            }
        
        distance_to_target = active_spot.get_position_error(car_x, car_y)
        orientation_error = active_spot.get_orientation_error(car_angle)
        
        # Calculate progress score (0.0 to 1.0)
        # Based on normalized distance and orientation errors
        max_distance = math.sqrt(self.environment_width**2 + self.environment_height**2)
        distance_score = 1.0 - min(distance_to_target / max_distance, 1.0)
        orientation_score = 1.0 - min(orientation_error / math.pi, 1.0)
        
        # Combined progress score (weighted average)
        progress_score = 0.7 * distance_score + 0.3 * orientation_score
        
        return {
            'distance_to_target': distance_to_target,
            'orientation_error': orientation_error,
            'orientation_error_degrees': math.degrees(orientation_error),
            'distance_score': distance_score,
            'orientation_score': orientation_score,
            'progress_score': progress_score,
            'is_successful': active_spot.is_parking_successful(car_x, car_y, car_angle)
        }
    
    def get_detailed_accuracy(self, car_x: float, car_y: float, car_angle: float) -> Dict[str, Any]:
        """
        Get detailed parking accuracy for the active spot.
        
        Args:
            car_x, car_y: Car center position
            car_angle: Car orientation in radians
            
        Returns:
            Dictionary with detailed accuracy metrics
        """
        active_spot = self.get_active_spot()
        if active_spot is None:
            return {'error': 'No active parking spot'}
        
        return active_spot.get_parking_accuracy(car_x, car_y, car_angle)
    
    def _is_spot_in_bounds(self, spot: ParkingSpot) -> bool:
        """Check if parking spot is within environment bounds."""
        corners = spot.get_corners()
        for x, y in corners:
            if x < 0 or x > self.environment_width or y < 0 or y > self.environment_height:
                return False
        return True
    
    def render_all_spots(self, surface: pygame.Surface, scale: float):
        """
        Render all parking spots.
        
        Args:
            surface: Pygame surface to draw on
            scale: Scale factor for coordinate conversion
        """
        for i, spot in enumerate(self.parking_spots):
            # Use different colors for active vs inactive spots
            if i == self.active_spot_index:
                color = (0, 255, 0)  # Green for active spot
            else:
                color = (100, 255, 100)  # Light green for inactive spots
            
            spot.render(surface, scale, color)
    
    def get_spots_summary(self) -> Dict[str, Any]:
        """Get summary information about all parking spots."""
        summary = {
            'total_spots': len(self.parking_spots),
            'active_spot_index': self.active_spot_index,
            'by_type': {},
            'spots': []
        }
        
        for i, spot in enumerate(self.parking_spots):
            spot_type = spot.spot_type.value
            if spot_type not in summary['by_type']:
                summary['by_type'][spot_type] = 0
            summary['by_type'][spot_type] += 1
            
            summary['spots'].append({
                'index': i,
                'type': spot_type,
                'position': (spot.x, spot.y),
                'angle_degrees': math.degrees(spot.angle),
                'size': (spot.length, spot.width),
                'is_active': i == self.active_spot_index
            })
        
        return summary
    
    def __str__(self) -> str:
        """String representation of parking spot manager."""
        summary = self.get_spots_summary()
        lines = [f"ParkingSpotManager: {summary['total_spots']} spots"]
        for spot_type, count in summary['by_type'].items():
            lines.append(f"  {spot_type}: {count}")
        if summary['total_spots'] > 0:
            lines.append(f"  Active spot: {summary['active_spot_index']}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representation of parking spot manager."""
        return f"ParkingSpotManager(spots={len(self.parking_spots)}, active={self.active_spot_index})"


# Global instance for easy access
default_parking_manager = None 