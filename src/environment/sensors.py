"""
Distance Sensor Implementation
Simulates 8-directional distance sensors (ultrasonic/LiDAR) for obstacle detection.

Based on the state space definition from the research paper:
s_t = [x_t, y_t, θ_t, v_t, d_1...d_8]^T

Where d_k is the distance reading input of the kth distance sensor (k = 8).
The sensors provide 360-degree coverage around the vehicle at 45-degree intervals.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import pygame


class DistanceSensor:
    """
    Individual distance sensor (ultrasonic/LiDAR simulation).
    
    Casts rays to detect distance to nearest obstacle in a specific direction.
    """
    
    def __init__(
        self, 
        relative_angle: float, 
        max_range: float = 20.0,
        resolution: float = 0.1
    ):
        """
        Initialize distance sensor.
        
        Args:
            relative_angle: Sensor angle relative to car orientation (radians)
            max_range: Maximum detection range (meters)
            resolution: Ray casting resolution (meters)
        """
        self.relative_angle = relative_angle
        self.max_range = max_range
        self.resolution = resolution
        self.last_reading = max_range  # Initialize to max range
        
    def get_reading(
        self, 
        car_x: float, 
        car_y: float, 
        car_theta: float,
        environment_bounds: Tuple[float, float, float, float],
        obstacles: List[Any] = None
    ) -> float:
        """
        Get distance reading from sensor.
        
        Args:
            car_x: Car X position
            car_y: Car Y position  
            car_theta: Car orientation angle (radians)
            environment_bounds: (min_x, min_y, max_x, max_y) environment boundaries
            obstacles: List of obstacle objects (for future use)
            
        Returns:
            Distance to nearest obstacle (meters), max_range if no obstacle
        """
        # Calculate global sensor angle
        global_angle = car_theta + self.relative_angle
        
        # Cast ray from car position
        distance = self._cast_ray(
            car_x, car_y, global_angle, 
            environment_bounds, obstacles
        )
        
        self.last_reading = distance
        return distance
    
    def _cast_ray(
        self,
        start_x: float,
        start_y: float, 
        angle: float,
        environment_bounds: Tuple[float, float, float, float],
        obstacles: List[Any] = None
    ) -> float:
        """
        Cast ray and find distance to nearest intersection.
        
        Args:
            start_x, start_y: Ray starting position
            angle: Ray direction (radians)
            environment_bounds: Environment boundaries
            obstacles: List of obstacles to check
            
        Returns:
            Distance to nearest intersection
        """
        min_x, min_y, max_x, max_y = environment_bounds
        
        # Ray direction
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # Find intersection with environment boundaries
        distances = []
        
        # Check intersection with left boundary (x = min_x)
        if dx != 0:
            t = (min_x - start_x) / dx
            if t > 0:
                y_intersect = start_y + t * dy
                if min_y <= y_intersect <= max_y:
                    distances.append(t)
        
        # Check intersection with right boundary (x = max_x)
        if dx != 0:
            t = (max_x - start_x) / dx
            if t > 0:
                y_intersect = start_y + t * dy
                if min_y <= y_intersect <= max_y:
                    distances.append(t)
        
        # Check intersection with bottom boundary (y = min_y)
        if dy != 0:
            t = (min_y - start_y) / dy
            if t > 0:
                x_intersect = start_x + t * dx
                if min_x <= x_intersect <= max_x:
                    distances.append(t)
        
        # Check intersection with top boundary (y = max_y)  
        if dy != 0:
            t = (max_y - start_y) / dy
            if t > 0:
                x_intersect = start_x + t * dx
                if min_x <= x_intersect <= max_x:
                    distances.append(t)
        
        # TODO: Add obstacle intersection checks when obstacles are implemented
        # This will be done in Phase 3
        
        # Return minimum distance, clamped to max range
        if distances:
            min_distance = min(distances)
            return min(min_distance, self.max_range)
        else:
            return self.max_range


class SensorArray:
    """
    Array of 8 distance sensors providing 360-degree coverage.
    
    Sensors are positioned at 45-degree intervals around the vehicle:
    - Sensor 0: Front (0°)
    - Sensor 1: Front-right (45°) 
    - Sensor 2: Right (90°)
    - Sensor 3: Rear-right (135°)
    - Sensor 4: Rear (180°)
    - Sensor 5: Rear-left (225°)
    - Sensor 6: Left (270°)
    - Sensor 7: Front-left (315°)
    """
    
    def __init__(self, max_range: float = 20.0, resolution: float = 0.1):
        """
        Initialize sensor array with 8 sensors at 45-degree intervals.
        
        Args:
            max_range: Maximum detection range in meters
            resolution: Distance measurement resolution
        """
        self.max_range = max_range
        self.resolution = resolution
        self.n_sensors = 8
        
        # Sensor directions: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        # (front, front-right, right, rear-right, rear, rear-left, left, front-left)
        self.sensor_angles = [
            0.0,                    # 0° - Front
            math.pi / 4,           # 45° - Front-right  
            math.pi / 2,           # 90° - Right
            3 * math.pi / 4,       # 135° - Rear-right
            math.pi,               # 180° - Rear
            5 * math.pi / 4,       # 225° - Rear-left
            3 * math.pi / 2,       # 270° - Left
            7 * math.pi / 4        # 315° - Front-left
        ]
        
        # Create sensors
        self.sensors = []
        for angle in self.sensor_angles:
            sensor = DistanceSensor(
                relative_angle=angle,
                max_range=max_range,
                resolution=resolution
            )
            self.sensors.append(sensor)
        
        # Sensor names for debugging and analysis
        self.sensor_names = [
            "Front", "Front-Right", "Right", "Rear-Right",
            "Rear", "Rear-Left", "Left", "Front-Left"
        ]
        
        # Last readings for state vector
        self.last_readings = [max_range] * self.n_sensors
    
    def get_all_readings(
        self,
        car_x: float,
        car_y: float, 
        car_theta: float,
        environment_bounds: Tuple[float, float, float, float],
        obstacles: List[Any] = None
    ) -> List[float]:
        """
        Get distance readings from all 8 sensors.
        
        Args:
            car_x: Car X position
            car_y: Car Y position
            car_theta: Car orientation angle (radians) 
            environment_bounds: Environment boundaries
            obstacles: List of obstacles
            
        Returns:
            List of 8 distance readings [d_1, d_2, ..., d_8]
        """
        readings = []
        
        for i, sensor in enumerate(self.sensors):
            distance = sensor.get_reading(
                car_x, car_y, car_theta,
                environment_bounds, obstacles
            )
            readings.append(distance)
        
        self.last_readings = readings
        return readings
    
    def get_reading(self, sensor_index: int) -> float:
        """Get last reading from specific sensor."""
        if 0 <= sensor_index < len(self.sensors):
            return self.last_readings[sensor_index]
        return self.max_range
    
    def get_front_sensors(self) -> Tuple[float, float, float]:
        """Get readings from front-facing sensors (front-left, front, front-right)."""
        return (self.last_readings[7], self.last_readings[0], self.last_readings[1])
    
    def get_rear_sensors(self) -> Tuple[float, float, float]:
        """Get readings from rear-facing sensors (rear-left, rear, rear-right)."""
        return (self.last_readings[5], self.last_readings[4], self.last_readings[3])
    
    def get_side_sensors(self) -> Tuple[float, float]:
        """Get readings from side sensors (left, right)."""
        return (self.last_readings[6], self.last_readings[2])
    
    def get_minimum_distance(self) -> float:
        """Get minimum distance from all sensors (closest obstacle)."""
        return min(self.last_readings)
    
    def get_sensor_directions(self) -> List[Tuple[float, float]]:
        """
        Get unit direction vectors for all sensors in global coordinates.
        
        Returns:
            List of (dx, dy) direction vectors for each sensor
        """
        directions = []
        for sensor in self.sensors:
            dx = math.cos(sensor.relative_angle)
            dy = math.sin(sensor.relative_angle)
            directions.append((dx, dy))
        return directions
    
    def normalize_readings(self, readings: List[float] = None) -> List[float]:
        """
        Normalize sensor readings to [0, 1] range.
        
        Args:
            readings: Optional specific readings to normalize, uses last_readings if None
            
        Returns:
            Normalized readings where 0 = obstacle at sensor, 1 = no obstacle in range
        """
        if readings is None:
            readings = self.last_readings
            
        normalized = []
        for reading in readings:
            normalized_value = reading / self.max_range
            normalized.append(min(1.0, max(0.0, normalized_value)))
        
        return normalized
    
    def detect_collision_risk(
        self, 
        threshold_distance: float = 2.0,
        car_velocity: float = 0.0
    ) -> Dict[str, Any]:
        """
        Analyze sensor readings for collision risk.
        
        Args:
            threshold_distance: Distance threshold for collision warning (meters)
            car_velocity: Current car velocity for directional analysis
            
        Returns:
            Dictionary with collision risk analysis
        """
        risk_analysis = {
            'has_risk': False,
            'risk_directions': [],
            'min_distance': self.get_minimum_distance(),
            'front_clear': True,
            'rear_clear': True,
            'sides_clear': True
        }
        
        # Check each sensor
        for i, reading in enumerate(self.last_readings):
            if reading < threshold_distance:
                risk_analysis['has_risk'] = True
                risk_analysis['risk_directions'].append({
                    'sensor': i,
                    'direction': self.sensor_names[i],
                    'distance': reading
                })
        
        # Check specific regions
        front_distances = self.get_front_sensors()
        rear_distances = self.get_rear_sensors()
        side_distances = self.get_side_sensors()
        
        risk_analysis['front_clear'] = all(d > threshold_distance for d in front_distances)
        risk_analysis['rear_clear'] = all(d > threshold_distance for d in rear_distances)
        risk_analysis['sides_clear'] = all(d > threshold_distance for d in side_distances)
        
        return risk_analysis
    
    def visualize_sensors(
        self, 
        surface: pygame.Surface,
        car_x: float, 
        car_y: float,
        car_theta: float,
        scale: float = 1.0,
        show_rays: bool = True
    ):
        """
        Visualize sensor readings on pygame surface.
        
        Args:
            surface: Pygame surface to draw on
            car_x, car_y: Car position in world coordinates
            car_theta: Car orientation
            scale: Scale factor for coordinate conversion
            show_rays: Whether to draw sensor rays
        """
        if not show_rays:
            return
            
        # Convert world coordinates to screen coordinates
        screen_x = int(car_x * scale)
        screen_y = int(car_y * scale)
        
        for i, (sensor, reading) in enumerate(zip(self.sensors, self.last_readings)):
            # Calculate sensor ray endpoint
            global_angle = car_theta + sensor.relative_angle
            end_x = car_x + reading * math.cos(global_angle)
            end_y = car_y + reading * math.sin(global_angle)
            
            screen_end_x = int(end_x * scale)
            screen_end_y = int(end_y * scale)
            
            # Color based on distance (red = close, green = far)
            color_intensity = min(255, int(255 * reading / self.max_range))
            color = (255 - color_intensity, color_intensity, 0)
            
            # Draw sensor ray
            pygame.draw.line(surface, color, (screen_x, screen_y), (screen_end_x, screen_end_y), 1)
            
            # Draw endpoint marker
            pygame.draw.circle(surface, color, (screen_end_x, screen_end_y), 2)
    
    def get_state_vector_component(self) -> List[float]:
        """
        Get sensor readings as component for state vector.
        
        Returns:
            List of 8 distance readings for state vector: [d_1, d_2, ..., d_8]
        """
        return self.last_readings.copy()
    
    def __str__(self) -> str:
        """String representation of sensor array."""
        lines = ["Distance Sensor Array (8 sensors):"]
        for i, (name, reading) in enumerate(zip(self.sensor_names, self.last_readings)):
            lines.append(f"  Sensor {i} ({name:11s}): {reading:6.2f}m")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representation of sensor array."""
        return f"SensorArray(n_sensors=8, max_range={self.max_range}m)"

    def get_distance_readings_with_obstacles(
        self,
        car_x: float,
        car_y: float,
        car_theta: float,
        obstacle_manager: Any
    ) -> List[float]:
        """
        Get distance readings using obstacle manager for Phase 3 integration.
        
        Args:
            car_x: Car X position
            car_y: Car Y position
            car_theta: Car orientation angle (radians)
            obstacle_manager: ObstacleManager instance for ray intersection
            
        Returns:
            List of 8 distance readings [d_1, d_2, ..., d_8]
        """
        readings = []
        
        for i, sensor in enumerate(self.sensors):
            # Calculate sensor direction in global coordinates
            global_angle = car_theta + sensor.relative_angle
            
            # Use obstacle manager's ray intersection method
            distance = obstacle_manager.get_ray_intersection(
                car_x, car_y, global_angle, self.max_range
            )
            
            readings.append(distance)
        
        self.last_readings = readings
        return readings
    
    def get_distance_readings(
        self,
        car_x: float,
        car_y: float,
        car_theta: float,
        environment_bounds: Tuple[float, float, float, float]
    ) -> List[float]:
        """
        Get distance readings using only environment boundaries (no obstacles).
        
        Args:
            car_x: Car X position
            car_y: Car Y position 
            car_theta: Car orientation angle (radians)
            environment_bounds: Environment boundaries (min_x, min_y, max_x, max_y)
            
        Returns:
            List of 8 distance readings [d_1, d_2, ..., d_8]
        """
        readings = []
        
        for i, sensor in enumerate(self.sensors):
            distance = sensor.get_reading(
                car_x, car_y, car_theta,
                environment_bounds, obstacles=None
            )
            readings.append(distance)
        
        self.last_readings = readings
        return readings


# Global instance for easy access
default_sensor_array = SensorArray() 