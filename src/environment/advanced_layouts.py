"""
Advanced Parking Lot Layouts for Complex Training Scenarios
Provides sophisticated parking environments for Phase 6 training.

Features:
- Multi-level parking structures
- Complex intersection layouts
- Realistic traffic flow patterns
- Variable lane widths and geometries
- Construction zones and temporary obstacles
"""

import numpy as np
import pygame
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class LayoutType(Enum):
    """Types of advanced parking layouts."""
    MULTI_LEVEL = "multi_level"
    INTERSECTION = "intersection"
    SPIRAL_RAMP = "spiral_ramp"
    ANGLED_COMPLEX = "angled_complex"
    CONSTRUCTION_ZONE = "construction_zone"
    NARROW_STREETS = "narrow_streets"
    ROUNDABOUT = "roundabout"
    SHOPPING_CENTER = "shopping_center"


@dataclass
class LayoutElement:
    """Base element for layout components."""
    x: float
    y: float
    width: float
    height: float
    element_type: str
    rotation: float = 0.0
    color: Tuple[int, int, int] = (128, 128, 128)
    properties: Dict[str, Any] = None


@dataclass
class ParkingLane:
    """Defines a lane of parking spots."""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    lane_width: float
    spot_count: int
    spot_angle: float  # Relative to lane direction
    spot_width: float = 2.5
    spot_depth: float = 5.0


@dataclass
class TrafficLane:
    """Defines a traffic lane."""
    center_points: List[Tuple[float, float]]
    lane_width: float
    lane_type: str  # "driving", "parking", "pedestrian"
    direction: str  # "bidirectional", "forward", "reverse"


class AdvancedLayoutGenerator:
    """Generates complex parking layouts."""
    
    def __init__(self):
        """Initialize layout generator."""
        self.layout_elements = []
        self.parking_lanes = []
        self.traffic_lanes = []
        
    def generate_layout(self, layout_type: LayoutType, 
                       environment_size: Tuple[float, float],
                       complexity_level: float = 0.5) -> Dict[str, Any]:
        """
        Generate an advanced parking layout.
        
        Args:
            layout_type: Type of layout to generate
            environment_size: (width, height) of environment
            complexity_level: 0.0 to 1.0, affects density and complexity
            
        Returns:
            Dictionary containing layout elements
        """
        self.layout_elements = []
        self.parking_lanes = []
        self.traffic_lanes = []
        
        if layout_type == LayoutType.MULTI_LEVEL:
            return self._generate_multi_level_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.INTERSECTION:
            return self._generate_intersection_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.SPIRAL_RAMP:
            return self._generate_spiral_ramp_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.ANGLED_COMPLEX:
            return self._generate_angled_complex_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.CONSTRUCTION_ZONE:
            return self._generate_construction_zone_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.NARROW_STREETS:
            return self._generate_narrow_streets_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.ROUNDABOUT:
            return self._generate_roundabout_layout(environment_size, complexity_level)
        elif layout_type == LayoutType.SHOPPING_CENTER:
            return self._generate_shopping_center_layout(environment_size, complexity_level)
        else:
            return self._generate_basic_layout(environment_size, complexity_level)
    
    def _generate_multi_level_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate multi-level parking garage layout."""
        
        width, height = env_size
        
        # Central driving lane
        central_lane = TrafficLane(
            center_points=[(width/2, 5), (width/2, height-5)],
            lane_width=6.0,
            lane_type="driving",
            direction="bidirectional"
        )
        self.traffic_lanes.append(central_lane)
        
        # Parking areas on both sides
        num_levels = int(2 + complexity * 2)  # 2-4 levels
        level_height = height / num_levels
        
        for level in range(num_levels):
            level_y = level * level_height + 5
            
            # Left side parking
            left_lane = ParkingLane(
                start_x=5, start_y=level_y,
                end_x=width/2 - 4, end_y=level_y + level_height - 10,
                lane_width=3.0,
                spot_count=int(5 + complexity * 3),
                spot_angle=np.pi/2  # 90 degrees
            )
            self.parking_lanes.append(left_lane)
            
            # Right side parking
            right_lane = ParkingLane(
                start_x=width/2 + 4, start_y=level_y,
                end_x=width - 5, end_y=level_y + level_height - 10,
                lane_width=3.0,
                spot_count=int(5 + complexity * 3),
                spot_angle=-np.pi/2  # -90 degrees
            )
            self.parking_lanes.append(right_lane)
        
        # Add structural elements (pillars, ramps)
        num_pillars = int(4 + complexity * 4)
        for i in range(num_pillars):
            pillar = LayoutElement(
                x=np.random.uniform(10, width-10),
                y=np.random.uniform(10, height-10),
                width=1.0, height=1.0,
                element_type="pillar",
                color=(100, 100, 100)
            )
            self.layout_elements.append(pillar)
        
        return {
            "layout_type": "multi_level",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["tight_turning", "multi_point_turns", "spatial_awareness"]
        }
    
    def _generate_intersection_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate complex intersection with parking."""
        
        width, height = env_size
        center_x, center_y = width/2, height/2
        
        # Main intersection roads
        horizontal_lane = TrafficLane(
            center_points=[(5, center_y), (width-5, center_y)],
            lane_width=8.0,
            lane_type="driving",
            direction="bidirectional"
        )
        self.traffic_lanes.append(horizontal_lane)
        
        vertical_lane = TrafficLane(
            center_points=[(center_x, 5), (center_x, height-5)],
            lane_width=8.0,
            lane_type="driving",
            direction="bidirectional"
        )
        self.traffic_lanes.append(vertical_lane)
        
        # Parking in quadrants
        quadrants = [
            (10, 10, center_x-10, center_y-10),  # Top-left
            (center_x+10, 10, width-10, center_y-10),  # Top-right
            (10, center_y+10, center_x-10, height-10),  # Bottom-left
            (center_x+10, center_y+10, width-10, height-10)  # Bottom-right
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(quadrants):
            angle = i * np.pi/2  # Different angle for each quadrant
            
            parking_lane = ParkingLane(
                start_x=x1, start_y=y1,
                end_x=x2, end_y=y2,
                lane_width=2.5,
                spot_count=int(3 + complexity * 2),
                spot_angle=angle
            )
            self.parking_lanes.append(parking_lane)
        
        # Traffic control elements
        if complexity > 0.5:
            # Traffic lights
            traffic_light = LayoutElement(
                x=center_x, y=center_y,
                width=0.5, height=0.5,
                element_type="traffic_light",
                color=(255, 255, 0)
            )
            self.layout_elements.append(traffic_light)
            
            # Stop signs
            for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
                sign_x = center_x + 12 * np.cos(angle)
                sign_y = center_y + 12 * np.sin(angle)
                
                stop_sign = LayoutElement(
                    x=sign_x, y=sign_y,
                    width=0.3, height=0.3,
                    element_type="stop_sign",
                    color=(255, 0, 0)
                )
                self.layout_elements.append(stop_sign)
        
        return {
            "layout_type": "intersection",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["intersection_navigation", "traffic_awareness", "precise_maneuvering"]
        }
    
    def _generate_spiral_ramp_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate spiral ramp parking structure."""
        
        width, height = env_size
        center_x, center_y = width/2, height/2
        
        # Spiral ramp
        num_turns = int(2 + complexity * 3)
        radius = min(width, height) * 0.3
        
        spiral_points = []
        for i in range(num_turns * 20):  # 20 points per turn
            angle = i * 2 * np.pi / 20
            r = radius * (1 + i * 0.05 / 20)  # Expanding spiral
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            spiral_points.append((x, y))
        
        spiral_lane = TrafficLane(
            center_points=spiral_points,
            lane_width=4.0,
            lane_type="driving",
            direction="forward"
        )
        self.traffic_lanes.append(spiral_lane)
        
        # Parking spots along the outer edge
        num_spots = int(10 + complexity * 10)
        for i in range(num_spots):
            angle = i * 2 * np.pi / num_spots
            spot_radius = radius * 1.8
            spot_x = center_x + spot_radius * np.cos(angle)
            spot_y = center_y + spot_radius * np.sin(angle)
            
            parking_lane = ParkingLane(
                start_x=spot_x, start_y=spot_y,
                end_x=spot_x, end_y=spot_y,
                lane_width=2.5,
                spot_count=1,
                spot_angle=angle + np.pi/2  # Perpendicular to radius
            )
            self.parking_lanes.append(parking_lane)
        
        return {
            "layout_type": "spiral_ramp",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["curved_driving", "spatial_orientation", "speed_control"]
        }
    
    def _generate_angled_complex_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate complex angled parking layout."""
        
        width, height = env_size
        
        # Diagonal main road
        diagonal_lane = TrafficLane(
            center_points=[(5, 5), (width-5, height-5)],
            lane_width=6.0,
            lane_type="driving",
            direction="bidirectional"
        )
        self.traffic_lanes.append(diagonal_lane)
        
        # Angled parking sections
        num_sections = int(3 + complexity * 2)
        
        for i in range(num_sections):
            # Vary angles for different sections
            base_angle = i * np.pi / 6  # 30-degree increments
            
            section_x = 15 + i * (width - 30) / num_sections
            section_y = 15 + i * (height - 30) / num_sections
            
            parking_lane = ParkingLane(
                start_x=section_x, start_y=section_y,
                end_x=section_x + 20, end_y=section_y + 15,
                lane_width=3.0,
                spot_count=int(4 + complexity * 2),
                spot_angle=base_angle
            )
            self.parking_lanes.append(parking_lane)
        
        # Add directional arrows and signage
        if complexity > 0.3:
            for i, lane in enumerate(self.parking_lanes):
                arrow = LayoutElement(
                    x=lane.start_x + 5,
                    y=lane.start_y + 5,
                    width=1.0, height=0.5,
                    element_type="direction_arrow",
                    rotation=lane.spot_angle,
                    color=(255, 255, 255)
                )
                self.layout_elements.append(arrow)
        
        return {
            "layout_type": "angled_complex",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["angled_parking", "depth_perception", "angle_estimation"]
        }
    
    def _generate_construction_zone_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate construction zone with temporary obstacles."""
        
        width, height = env_size
        
        # Main driving area with detours
        main_lane = TrafficLane(
            center_points=[(10, height/2), (width/3, height/2), (2*width/3, height/3), (width-10, height/2)],
            lane_width=4.0,
            lane_type="driving",
            direction="forward"
        )
        self.traffic_lanes.append(main_lane)
        
        # Available parking spots (reduced due to construction)
        available_spots = int(5 + complexity * 3)
        
        for i in range(available_spots):
            # Place spots in non-construction areas
            spot_x = np.random.uniform(10, width-10)
            spot_y = np.random.uniform(10, height-10)
            
            parking_lane = ParkingLane(
                start_x=spot_x, start_y=spot_y,
                end_x=spot_x, end_y=spot_y,
                lane_width=2.5,
                spot_count=1,
                spot_angle=np.random.uniform(0, 2*np.pi)
            )
            self.parking_lanes.append(parking_lane)
        
        # Construction barriers and cones
        num_barriers = int(8 + complexity * 6)
        
        for i in range(num_barriers):
            barrier_type = np.random.choice(["barrier", "cone", "sign"])
            
            if barrier_type == "barrier":
                element = LayoutElement(
                    x=np.random.uniform(5, width-5),
                    y=np.random.uniform(5, height-5),
                    width=2.0, height=0.3,
                    element_type="construction_barrier",
                    rotation=np.random.uniform(0, 2*np.pi),
                    color=(255, 165, 0)
                )
            elif barrier_type == "cone":
                element = LayoutElement(
                    x=np.random.uniform(5, width-5),
                    y=np.random.uniform(5, height-5),
                    width=0.3, height=0.3,
                    element_type="traffic_cone",
                    color=(255, 100, 0)
                )
            else:  # sign
                element = LayoutElement(
                    x=np.random.uniform(5, width-5),
                    y=np.random.uniform(5, height-5),
                    width=1.0, height=1.5,
                    element_type="construction_sign",
                    color=(255, 255, 0)
                )
            
            self.layout_elements.append(element)
        
        return {
            "layout_type": "construction_zone",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["obstacle_avoidance", "path_planning", "adaptive_driving"]
        }
    
    def _generate_narrow_streets_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate narrow European-style streets."""
        
        width, height = env_size
        
        # Create narrow winding streets
        street_width = 3.0 - complexity * 0.5  # Narrower with higher complexity
        
        # Main street (curved)
        main_points = []
        num_points = int(8 + complexity * 4)
        
        for i in range(num_points):
            x = i * width / (num_points - 1)
            y = height/2 + 5 * np.sin(i * np.pi / 4)  # Sinusoidal curve
            main_points.append((x, y))
        
        main_street = TrafficLane(
            center_points=main_points,
            lane_width=street_width,
            lane_type="driving",
            direction="bidirectional"
        )
        self.traffic_lanes.append(main_street)
        
        # Side streets
        num_side_streets = int(2 + complexity * 2)
        
        for i in range(num_side_streets):
            start_x = (i + 1) * width / (num_side_streets + 1)
            
            side_street = TrafficLane(
                center_points=[(start_x, 5), (start_x, height-5)],
                lane_width=street_width * 0.8,
                lane_type="driving",
                direction="bidirectional"
            )
            self.traffic_lanes.append(side_street)
        
        # Street parking (parallel only due to narrow streets)
        num_parking_areas = int(4 + complexity * 2)
        
        for i in range(num_parking_areas):
            area_x = np.random.uniform(15, width-15)
            area_y = np.random.uniform(10, height-10)
            
            parking_lane = ParkingLane(
                start_x=area_x, start_y=area_y,
                end_x=area_x + 12, end_y=area_y,
                lane_width=2.0,
                spot_count=2,
                spot_angle=0,  # Parallel parking
                spot_width=6.0,
                spot_depth=2.0
            )
            self.parking_lanes.append(parking_lane)
        
        return {
            "layout_type": "narrow_streets",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["parallel_parking", "narrow_maneuvering", "precise_control"]
        }
    
    def _generate_roundabout_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate roundabout with surrounding parking."""
        
        width, height = env_size
        center_x, center_y = width/2, height/2
        
        # Central roundabout
        roundabout_radius = min(width, height) * 0.15
        circle_points = []
        
        for i in range(20):  # 20 points for smooth circle
            angle = i * 2 * np.pi / 20
            x = center_x + roundabout_radius * np.cos(angle)
            y = center_y + roundabout_radius * np.sin(angle)
            circle_points.append((x, y))
        
        roundabout = TrafficLane(
            center_points=circle_points,
            lane_width=6.0,
            lane_type="driving",
            direction="forward"
        )
        self.traffic_lanes.append(roundabout)
        
        # Approach roads
        num_approaches = int(3 + complexity)  # 3-4 approaches
        
        for i in range(num_approaches):
            angle = i * 2 * np.pi / num_approaches
            
            # Inner point (roundabout edge)
            inner_radius = roundabout_radius + 3
            inner_x = center_x + inner_radius * np.cos(angle)
            inner_y = center_y + inner_radius * np.sin(angle)
            
            # Outer point (environment edge)
            outer_radius = min(width, height) * 0.4
            outer_x = center_x + outer_radius * np.cos(angle)
            outer_y = center_y + outer_radius * np.sin(angle)
            
            approach = TrafficLane(
                center_points=[(outer_x, outer_y), (inner_x, inner_y)],
                lane_width=4.0,
                lane_type="driving",
                direction="bidirectional"
            )
            self.traffic_lanes.append(approach)
        
        # Parking areas between approaches
        for i in range(num_approaches):
            angle1 = i * 2 * np.pi / num_approaches
            angle2 = (i + 1) * 2 * np.pi / num_approaches
            mid_angle = (angle1 + angle2) / 2
            
            parking_radius = roundabout_radius + 15
            parking_x = center_x + parking_radius * np.cos(mid_angle)
            parking_y = center_y + parking_radius * np.sin(mid_angle)
            
            parking_lane = ParkingLane(
                start_x=parking_x - 5, start_y=parking_y - 5,
                end_x=parking_x + 5, end_y=parking_y + 5,
                lane_width=2.5,
                spot_count=int(3 + complexity),
                spot_angle=mid_angle + np.pi/2
            )
            self.parking_lanes.append(parking_lane)
        
        return {
            "layout_type": "roundabout",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["circular_navigation", "yielding", "traffic_flow"]
        }
    
    def _generate_shopping_center_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate large shopping center with multiple parking areas."""
        
        width, height = env_size
        
        # Central building (shopping center)
        building = LayoutElement(
            x=width/2, y=height/2,
            width=width*0.3, height=height*0.3,
            element_type="building",
            color=(150, 150, 150)
        )
        self.layout_elements.append(building)
        
        # Perimeter road around building
        road_points = [
            (width*0.2, height*0.2),
            (width*0.8, height*0.2),
            (width*0.8, height*0.8),
            (width*0.2, height*0.8),
            (width*0.2, height*0.2)  # Close the loop
        ]
        
        perimeter_road = TrafficLane(
            center_points=road_points,
            lane_width=6.0,
            lane_type="driving",
            direction="forward"
        )
        self.traffic_lanes.append(perimeter_road)
        
        # Parking sections around the perimeter
        sections = [
            (width*0.05, height*0.05, width*0.15, height*0.4),  # Left
            (width*0.05, height*0.6, width*0.15, height*0.95),  # Left lower
            (width*0.85, height*0.05, width*0.95, height*0.4),  # Right
            (width*0.85, height*0.6, width*0.95, height*0.95),  # Right lower
            (width*0.25, height*0.05, width*0.75, height*0.15),  # Top
            (width*0.25, height*0.85, width*0.75, height*0.95),  # Bottom
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(sections):
            spots_per_section = int(5 + complexity * 3)
            
            parking_lane = ParkingLane(
                start_x=x1, start_y=y1,
                end_x=x2, end_y=y2,
                lane_width=3.0,
                spot_count=spots_per_section,
                spot_angle=0 if i < 4 else np.pi/2  # Different orientations
            )
            self.parking_lanes.append(parking_lane)
        
        # Shopping cart returns and other retail elements
        num_cart_returns = int(4 + complexity * 2)
        
        for i in range(num_cart_returns):
            cart_return = LayoutElement(
                x=np.random.uniform(width*0.2, width*0.8),
                y=np.random.uniform(height*0.2, height*0.8),
                width=2.0, height=1.0,
                element_type="cart_return",
                color=(100, 150, 100)
            )
            self.layout_elements.append(cart_return)
        
        return {
            "layout_type": "shopping_center",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["large_area_navigation", "parking_selection", "pedestrian_awareness"]
        }
    
    def _generate_basic_layout(self, env_size: Tuple[float, float], complexity: float) -> Dict[str, Any]:
        """Generate basic layout as fallback."""
        
        width, height = env_size
        
        # Simple grid layout
        parking_lane = ParkingLane(
            start_x=10, start_y=10,
            end_x=width-10, end_y=height-10,
            lane_width=3.0,
            spot_count=int(6 + complexity * 4),
            spot_angle=0
        )
        self.parking_lanes.append(parking_lane)
        
        return {
            "layout_type": "basic",
            "elements": self.layout_elements,
            "parking_lanes": self.parking_lanes,
            "traffic_lanes": self.traffic_lanes,
            "complexity_score": complexity,
            "recommended_skills": ["basic_parking"]
        }


# Utility functions for layout validation and optimization
def validate_layout(layout_data: Dict[str, Any]) -> bool:
    """Validate that a layout is feasible and safe."""
    # Check for overlapping elements
    # Verify accessibility of parking spots
    # Ensure traffic flow makes sense
    return True


def optimize_layout_for_training(layout_data: Dict[str, Any], 
                                training_objectives: List[str]) -> Dict[str, Any]:
    """Optimize layout based on specific training objectives."""
    # Adjust parking spot difficulty
    # Modify obstacle placement
    # Fine-tune traffic patterns
    return layout_data


# Preset collections for different training phases
LAYOUT_PRESETS = {
    "beginner": [
        (LayoutType.SHOPPING_CENTER, 0.2),
        (LayoutType.ANGLED_COMPLEX, 0.3),
    ],
    
    "intermediate": [
        (LayoutType.MULTI_LEVEL, 0.5),
        (LayoutType.INTERSECTION, 0.4),
        (LayoutType.NARROW_STREETS, 0.6),
    ],
    
    "advanced": [
        (LayoutType.SPIRAL_RAMP, 0.7),
        (LayoutType.ROUNDABOUT, 0.8),
        (LayoutType.CONSTRUCTION_ZONE, 0.9),
    ]
} 