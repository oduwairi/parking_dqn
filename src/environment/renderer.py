"""
Parking Environment Renderer - Enhanced for Phase 3
Visualizes the 2D parking environment using pygame with obstacles, collision detection, and enhanced sensors.

Phase 3 Enhancements:
- Obstacle rendering (barriers, vehicles, pillars)
- Parking spot visualization with tolerances
- Sensor ray visualization
- Collision information display
"""

import pygame
import numpy as np
import math
from typing import Optional, Tuple, List, Any

# Color definitions (RGB)
COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'gray': (128, 128, 128),
    'light_gray': (200, 200, 200),
    'dark_gray': (64, 64, 64),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128),
    'cyan': (0, 255, 255),
    'pink': (255, 192, 203),
    'brown': (139, 69, 19),
}


class ParkingRenderer:
    """
    Enhanced renderer for parking environment with Phase 3 features.
    
    Supports rendering of obstacles, enhanced parking spots, sensor visualization,
    and collision information.
    """
    
    def __init__(
        self, 
        env_width: float, 
        env_height: float, 
        window_width: int = 800, 
        window_height: int = 600
    ):
        """
        Initialize enhanced renderer.
        
        Args:
            env_width: Environment width in meters
            env_height: Environment height in meters  
            window_width: Rendering window width in pixels
            window_height: Rendering window height in pixels
        """
        self.env_width = env_width
        self.env_height = env_height
        self.window_width = window_width
        self.window_height = window_height
        
        # Calculate scaling factors
        self.scale_x = window_width / env_width
        self.scale_y = window_height / env_height
        self.scale = min(self.scale_x, self.scale_y)  # Maintain aspect ratio
        
        # Center the environment in the window
        self.offset_x = (window_width - env_width * self.scale) / 2
        self.offset_y = (window_height - env_height * self.scale) / 2
        
        # Pygame initialization
        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int((self.env_height - y) * self.scale + self.offset_y)  # Flip Y axis
        return screen_x, screen_y
        
    def render(
        self, 
        car, 
        target_x: float, 
        target_y: float, 
        target_theta: float, 
        mode: str = 'human',
        # Phase 3: Enhanced parameters with defaults for backward compatibility
        sensor_readings: Optional[List[float]] = None,
        obstacles: Optional[List[Any]] = None,
        parking_spots: Optional[List[Any]] = None,
        collision_info: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """
        Enhanced render method with Phase 3 features.
        
        Args:
            car: CarAgent instance
            target_x: Target parking spot x coordinate
            target_y: Target parking spot y coordinate
            target_theta: Target orientation
            mode: Render mode ('human' or 'rgb_array')
            sensor_readings: Optional list of 8 sensor distance readings
            obstacles: Optional list of obstacle objects
            parking_spots: Optional list of parking spot objects
            collision_info: Optional collision information
            
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        if self.screen is None:
            pygame.display.init()
            pygame.display.set_caption("Autonomous Parking DQN - Phase 3")
            if mode == 'human':
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            else:
                self.screen = pygame.Surface((self.window_width, self.window_height))
                
        # Clear screen
        self.screen.fill(COLORS['light_gray'])
        
        # Draw environment boundaries
        self._draw_boundaries()
        
        # Phase 3: Draw obstacles first (so car renders on top)
        if obstacles:
            self._draw_obstacles(obstacles)
        
        # Phase 3: Draw enhanced parking spots
        if parking_spots:
            self._draw_parking_spots(parking_spots)
        else:
            # Fallback to simple parking spot for backward compatibility
            self._draw_parking_spot(target_x, target_y, target_theta)
        
        # Phase 3: Draw sensor rays
        if sensor_readings is not None:
            self._draw_sensor_rays(car, sensor_readings)
        
        # Draw car
        self._draw_car(car)
        
        # Phase 3: Draw collision information
        if collision_info:
            self._draw_collision_info(collision_info)
        
        # Draw enhanced info text
        self._draw_enhanced_info(car, target_x, target_y, collision_info, sensor_readings)
        
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
            return None
        else:
            # Return RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def _draw_boundaries(self):
        """Draw environment boundaries."""
        # Draw border
        border_points = [
            self.world_to_screen(0, 0),
            self.world_to_screen(self.env_width, 0),
            self.world_to_screen(self.env_width, self.env_height),
            self.world_to_screen(0, self.env_height)
        ]
        pygame.draw.polygon(self.screen, COLORS['black'], border_points, 3)
        
        # Draw grid lines (optional, for reference)
        grid_spacing = 5.0  # 5 meter grid
        for x in np.arange(0, self.env_width + grid_spacing, grid_spacing):
            start = self.world_to_screen(x, 0)
            end = self.world_to_screen(x, self.env_height)
            pygame.draw.line(self.screen, COLORS['gray'], start, end, 1)
            
        for y in np.arange(0, self.env_height + grid_spacing, grid_spacing):
            start = self.world_to_screen(0, y)
            end = self.world_to_screen(self.env_width, y)
            pygame.draw.line(self.screen, COLORS['gray'], start, end, 1)
    
    def _draw_obstacles(self, obstacles: List[Any]):
        """Draw static obstacles with Phase 3 support."""
        for obstacle in obstacles:
            # Get obstacle type to determine rendering
            obstacle_type = obstacle.obstacle_type.value
            
            if obstacle_type == "circular":
                # Draw circular obstacle
                center = self.world_to_screen(obstacle.x, obstacle.y)
                radius = int(obstacle.radius * self.scale)
                pygame.draw.circle(self.screen, obstacle.color, center, radius)
                pygame.draw.circle(self.screen, COLORS['black'], center, radius, 2)
                
            else:
                # Draw rectangular obstacles (barriers, vehicles, etc.)
                corners = obstacle.get_corners()
                screen_corners = [self.world_to_screen(cx, cy) for cx, cy in corners]
                pygame.draw.polygon(self.screen, obstacle.color, screen_corners)
                pygame.draw.polygon(self.screen, COLORS['black'], screen_corners, 2)
                
                # Add type label for clarity
                center = self.world_to_screen(obstacle.x, obstacle.y)
                if obstacle_type == "vehicle":
                    label = "CAR"
                elif obstacle_type == "barrier":
                    label = "WALL"
                else:
                    label = obstacle_type.upper()[:4]
                    
                text_surface = self.small_font.render(label, True, COLORS['white'])
                text_rect = text_surface.get_rect(center=center)
                self.screen.blit(text_surface, text_rect)
    
    def _draw_parking_spots(self, parking_spots: List[Any]):
        """Draw enhanced parking spots with tolerances."""
        for i, spot in enumerate(parking_spots):
            # Determine color based on active status
            if hasattr(spot, 'is_active') and spot.is_active:
                color = COLORS['green']
                tolerance_color = COLORS['yellow']
            else:
                color = COLORS['cyan']
                tolerance_color = COLORS['light_gray']
            
            # Draw main parking spot
            corners = spot.get_corners()
            screen_corners = [self.world_to_screen(cx, cy) for cx, cy in corners]
            pygame.draw.polygon(self.screen, color, screen_corners, 3)
            
            # Draw tolerance zone
            tolerance_radius = int(spot.position_tolerance * self.scale)
            center = self.world_to_screen(spot.x, spot.y)
            pygame.draw.circle(self.screen, tolerance_color, center, tolerance_radius, 1)
            
            # Draw orientation indicator
            arrow_length = min(spot.width, spot.length) * 0.4
            arrow_end_x = spot.x + arrow_length * math.cos(spot.angle)
            arrow_end_y = spot.y + arrow_length * math.sin(spot.angle)
            
            start_pos = self.world_to_screen(spot.x, spot.y)
            end_pos = self.world_to_screen(arrow_end_x, arrow_end_y)
            pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
            
            # Draw center point
            pygame.draw.circle(self.screen, color, center, 5)
            
            # Draw spot type label
            spot_type = spot.spot_type.value.upper()[:4]
            text_surface = self.small_font.render(f"P{i}:{spot_type}", True, color)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (center[0] + 10, center[1] - 10)
            self.screen.blit(text_surface, text_rect)
    
    def _draw_sensor_rays(self, car, sensor_readings: List[float]):
        """Draw sensor rays with distance visualization."""
        # Sensor angles (8 directions at 45° intervals)
        sensor_angles = [i * math.pi / 4 for i in range(8)]
        
        car_center = self.world_to_screen(car.x, car.y)
        
        for i, (angle_offset, distance) in enumerate(zip(sensor_angles, sensor_readings)):
            # Calculate global sensor angle
            global_angle = car.theta + angle_offset
            
            # Calculate ray endpoint
            end_x = car.x + distance * math.cos(global_angle)
            end_y = car.y + distance * math.sin(global_angle)
            end_pos = self.world_to_screen(end_x, end_y)
            
            # Color based on distance (red = close, green = far)
            max_range = 20.0  # Assume max sensor range
            color_intensity = min(255, int(255 * distance / max_range))
            if distance < 2.0:  # Collision warning
                ray_color = COLORS['red']
            elif distance < 5.0:  # Caution
                ray_color = COLORS['orange'] 
            else:  # Safe
                ray_color = (255 - color_intensity, color_intensity, 0)
            
            # Draw sensor ray
            pygame.draw.line(self.screen, ray_color, car_center, end_pos, 2)
            
            # Draw endpoint marker
            pygame.draw.circle(self.screen, ray_color, end_pos, 3)
    
    def _draw_collision_info(self, collision_info):
        """Draw collision information visualization."""
        if collision_info and hasattr(collision_info, 'collision_type'):
            collision_type = collision_info.collision_type.value
            
            if collision_type != "no_collision":
                # Draw collision point
                if hasattr(collision_info, 'collision_point'):
                    collision_pos = self.world_to_screen(
                        collision_info.collision_point[0], 
                        collision_info.collision_point[1]
                    )
                    # Flash red circle at collision point
                    pygame.draw.circle(self.screen, COLORS['red'], collision_pos, 10, 3)
                    pygame.draw.circle(self.screen, COLORS['yellow'], collision_pos, 15, 2)
                
                # Draw collision warning text
                warning_text = f"COLLISION: {collision_type.upper()}"
                text_surface = self.font.render(warning_text, True, COLORS['red'])
                self.screen.blit(text_surface, (10, 10))
            
    def _draw_parking_spot(self, x: float, y: float, theta: float):
        """Draw simple target parking spot (backward compatibility)."""
        # Parking spot dimensions
        spot_length = 6.0
        spot_width = 3.0
        
        # Calculate corner points
        corners = self._get_rectangle_corners(x, y, theta, spot_length, spot_width)
        screen_corners = [self.world_to_screen(cx, cy) for cx, cy in corners]
        
        # Draw parking spot outline
        pygame.draw.polygon(self.screen, COLORS['green'], screen_corners, 3)
        
        # Draw target orientation arrow
        arrow_length = 2.0
        arrow_end_x = x + arrow_length * math.cos(theta)
        arrow_end_y = y + arrow_length * math.sin(theta)
        
        start_pos = self.world_to_screen(x, y)
        end_pos = self.world_to_screen(arrow_end_x, arrow_end_y)
        pygame.draw.line(self.screen, COLORS['green'], start_pos, end_pos, 3)
        
        # Draw center point
        center_pos = self.world_to_screen(x, y)
        pygame.draw.circle(self.screen, COLORS['green'], center_pos, 5)
        
    def _draw_car(self, car):
        """Draw the car agent."""
        # Get car corners
        corners = car.get_corners()
        screen_corners = [self.world_to_screen(cx, cy) for cx, cy in corners]
        
        # Draw car body
        pygame.draw.polygon(self.screen, COLORS['blue'], screen_corners)
        pygame.draw.polygon(self.screen, COLORS['black'], screen_corners, 2)
        
        # Draw car direction arrow
        arrow_length = 1.5
        front_x = car.x + arrow_length * math.cos(car.theta)
        front_y = car.y + arrow_length * math.sin(car.theta)
        
        start_pos = self.world_to_screen(car.x, car.y)
        end_pos = self.world_to_screen(front_x, front_y)
        pygame.draw.line(self.screen, COLORS['red'], start_pos, end_pos, 3)
        
        # Draw center point
        center_pos = self.world_to_screen(car.x, car.y)
        pygame.draw.circle(self.screen, COLORS['red'], center_pos, 4)
        
        # Draw velocity vector (if moving)
        if abs(car.velocity) > 0.1:
            vel_length = min(abs(car.velocity) * 2.0, 3.0)  # Scale velocity for display
            vel_end_x = car.x + vel_length * math.cos(car.theta) * (1 if car.velocity > 0 else -1)
            vel_end_y = car.y + vel_length * math.sin(car.theta) * (1 if car.velocity > 0 else -1)
            
            vel_end_pos = self.world_to_screen(vel_end_x, vel_end_y)
            pygame.draw.line(self.screen, COLORS['purple'], start_pos, vel_end_pos, 2)
            
    def _draw_enhanced_info(self, car, target_x: float, target_y: float, 
                          collision_info=None, sensor_readings=None):
        """Draw enhanced information panel with Phase 3 data."""
        info_lines = []
        
        # Car state
        info_lines.append(f"Car: ({car.x:.1f}, {car.y:.1f}) θ={math.degrees(car.theta):.1f}°")
        info_lines.append(f"Velocity: {car.velocity:.2f} m/s")
        
        # Target info
        distance = math.sqrt((car.x - target_x)**2 + (car.y - target_y)**2)
        info_lines.append(f"Target: ({target_x:.1f}, {target_y:.1f})")
        info_lines.append(f"Distance: {distance:.2f}m")
        
        # Sensor info
        if sensor_readings:
            min_distance = min(sensor_readings)
            info_lines.append(f"Min Sensor: {min_distance:.2f}m")
            
        # Collision info
        if collision_info and hasattr(collision_info, 'collision_type'):
            collision_type = collision_info.collision_type.value
            if collision_type != "no_collision":
                info_lines.append(f"Status: {collision_type.upper()}")
            else:
                info_lines.append("Status: OK")
        
        # Render info text
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, COLORS['black'])
            self.screen.blit(text_surface, (10, 40 + i * 25))
        
        # Phase 3: Add legend
        legend_y = self.window_height - 120
        legend_items = [
            ("Car", COLORS['blue']),
            ("Target", COLORS['green']),
            ("Obstacles", COLORS['brown']),
            ("Sensors", COLORS['orange'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            pygame.draw.circle(self.screen, color, (20, legend_y + i * 20), 8)
            text_surface = self.small_font.render(label, True, COLORS['black'])
            self.screen.blit(text_surface, (35, legend_y + i * 20 - 8))

    def _draw_info(self, car, target_x: float, target_y: float):
        """Original info drawing method (backward compatibility)."""
        # Car position and orientation
        car_info = f"Car: ({car.x:.1f}, {car.y:.1f}) θ={math.degrees(car.theta):.1f}°"
        
        # Distance to target
        distance = math.sqrt((car.x - target_x)**2 + (car.y - target_y)**2)
        distance_info = f"Distance to target: {distance:.2f}m"
        
        # Velocity
        velocity_info = f"Velocity: {car.velocity:.2f} m/s"
        
        # Render text
        y_offset = 40
        for text in [car_info, distance_info, velocity_info]:
            text_surface = self.font.render(text, True, COLORS['black'])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

    def _get_rectangle_corners(
        self, 
        x: float, 
        y: float, 
        theta: float, 
        length: float, 
        width: float
    ) -> list:
        """Get corners of a rectangle centered at (x, y) with given orientation."""
        # Local corners
        local_corners = [
            (-length/2, -width/2),  # Rear left
            (+length/2, -width/2),  # Front left
            (+length/2, +width/2),  # Front right
            (-length/2, +width/2),  # Rear right
        ]
        
        # Rotate and translate
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        global_corners = []
        for lx, ly in local_corners:
            gx = x + lx * cos_theta - ly * sin_theta
            gy = y + lx * sin_theta + ly * cos_theta
            global_corners.append((gx, gy))
            
        return global_corners
        
    def close(self):
        """Clean up pygame resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None 