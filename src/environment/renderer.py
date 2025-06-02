"""
Parking Environment Renderer
Visualizes the 2D parking environment using pygame.
"""

import pygame
import numpy as np
import math
from typing import Optional, Tuple

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
}


class ParkingRenderer:
    """
    Renders the parking environment with car, target, and obstacles.
    """
    
    def __init__(
        self, 
        env_width: float, 
        env_height: float, 
        window_width: int = 800, 
        window_height: int = 600
    ):
        """
        Initialize renderer.
        
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
        mode: str = 'human'
    ) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            car: CarAgent instance
            target_x: Target parking spot x coordinate
            target_y: Target parking spot y coordinate
            target_theta: Target orientation
            mode: Render mode ('human' or 'rgb_array')
            
        Returns:
            RGB array if mode is 'rgb_array', None otherwise
        """
        if self.screen is None:
            pygame.display.init()
            pygame.display.set_caption("Autonomous Parking DQN")
            if mode == 'human':
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            else:
                self.screen = pygame.Surface((self.window_width, self.window_height))
                
        # Clear screen
        self.screen.fill(COLORS['light_gray'])
        
        # Draw environment boundaries
        self._draw_boundaries()
        
        # Draw parking spot
        self._draw_parking_spot(target_x, target_y, target_theta)
        
        # Draw car
        self._draw_car(car)
        
        # Draw info text
        self._draw_info(car, target_x, target_y)
        
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
            
    def _draw_parking_spot(self, x: float, y: float, theta: float):
        """Draw target parking spot."""
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
            vel_scale = 2.0
            vel_end_x = car.x + vel_scale * car.velocity * math.cos(car.theta)
            vel_end_y = car.y + vel_scale * car.velocity * math.sin(car.theta)
            
            vel_end_pos = self.world_to_screen(vel_end_x, vel_end_y)
            pygame.draw.line(self.screen, COLORS['orange'], start_pos, vel_end_pos, 2)
            
    def _draw_info(self, car, target_x: float, target_y: float):
        """Draw information text."""
        info_lines = [
            f"Position: ({car.x:.1f}, {car.y:.1f})",
            f"Orientation: {math.degrees(car.theta):.1f}°",
            f"Velocity: {car.velocity:.2f} m/s",
            f"Target: ({target_x:.1f}, {target_y:.1f})",
            f"Distance: {car.distance_to(target_x, target_y):.2f}m",
        ]
        
        y_offset = 10
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, COLORS['black'])
            self.screen.blit(text_surface, (10, y_offset + i * 25))
            
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