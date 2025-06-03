"""
Scenario Generator for Diverse Parking Training Environments
Creates varied parking scenarios for robust agent training in Phase 6.

Features:
- Multiple parking lot layouts
- Varying obstacle configurations
- Different parking spot sizes and orientations
- Weather and lighting conditions simulation
- Scenario difficulty progression
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ScenarioDifficulty(Enum):
    """Difficulty levels for parking scenarios."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class WeatherCondition(Enum):
    """Weather conditions affecting visibility and sensor range."""
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    FOG = "fog"
    SNOW = "snow"


@dataclass
class ParkingSpot:
    """Definition of a parking spot."""
    x: float
    y: float
    width: float
    height: float
    angle: float  # Rotation angle in radians
    spot_type: str  # "parallel", "perpendicular", "angled"
    difficulty_modifier: float = 1.0


@dataclass
class StaticObstacle:
    """Static obstacle definition."""
    x: float
    y: float
    width: float
    height: float
    obstacle_type: str  # "wall", "bollard", "vehicle", "barrier"


@dataclass
class ScenarioConfig:
    """Complete scenario configuration."""
    name: str
    difficulty: ScenarioDifficulty
    environment_size: Tuple[float, float]
    parking_spots: List[ParkingSpot]
    static_obstacles: List[StaticObstacle]
    weather: WeatherCondition
    time_of_day: str  # "day", "night", "dawn", "dusk"
    dynamic_obstacle_density: str  # "none", "light", "moderate", "heavy"
    sensor_noise_level: float  # 0.0 to 1.0
    max_episode_steps: int
    success_tolerance: Dict[str, float]  # position and angle tolerances


class ScenarioGenerator:
    """Generates diverse parking scenarios for training."""
    
    def __init__(self, random_seed: int = None):
        """Initialize scenario generator."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.scenario_templates = self._load_scenario_templates()
    
    def generate_scenario(self, difficulty: ScenarioDifficulty = None, 
                         scenario_type: str = "random") -> ScenarioConfig:
        """
        Generate a parking scenario.
        
        Args:
            difficulty: Target difficulty level
            scenario_type: Type of scenario to generate
            
        Returns:
            Complete scenario configuration
        """
        if difficulty is None:
            difficulty = random.choice(list(ScenarioDifficulty))
        
        if scenario_type == "random":
            scenario_type = random.choice([
                "supermarket", "street_parking", "garage", "mall", "residential",
                "highway_rest", "airport", "hospital", "school", "office"
            ])
        
        # Generate base scenario
        base_config = self._generate_base_scenario(scenario_type, difficulty)
        
        # Add environmental variations
        base_config = self._add_environmental_variations(base_config, difficulty)
        
        # Add dynamic elements based on difficulty
        base_config = self._add_dynamic_elements(base_config, difficulty)
        
        return base_config
    
    def _generate_base_scenario(self, scenario_type: str, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate base scenario layout."""
        
        if scenario_type == "supermarket":
            return self._generate_supermarket_scenario(difficulty)
        elif scenario_type == "street_parking":
            return self._generate_street_parking_scenario(difficulty)
        elif scenario_type == "garage":
            return self._generate_garage_scenario(difficulty)
        elif scenario_type == "mall":
            return self._generate_mall_scenario(difficulty)
        elif scenario_type == "residential":
            return self._generate_residential_scenario(difficulty)
        elif scenario_type == "highway_rest":
            return self._generate_highway_rest_scenario(difficulty)
        elif scenario_type == "airport":
            return self._generate_airport_scenario(difficulty)
        elif scenario_type == "hospital":
            return self._generate_hospital_scenario(difficulty)
        elif scenario_type == "school":
            return self._generate_school_scenario(difficulty)
        elif scenario_type == "office":
            return self._generate_office_scenario(difficulty)
        else:
            # Default to supermarket
            return self._generate_supermarket_scenario(difficulty)
    
    def _generate_supermarket_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate supermarket parking lot scenario."""
        
        # Environment size based on difficulty
        if difficulty == ScenarioDifficulty.EASY:
            env_size = (40.0, 25.0)
            num_spots = 3
            obstacle_density = 0.1
        elif difficulty == ScenarioDifficulty.MEDIUM:
            env_size = (50.0, 30.0)
            num_spots = 5
            obstacle_density = 0.2
        elif difficulty == ScenarioDifficulty.HARD:
            env_size = (60.0, 35.0)
            num_spots = 8
            obstacle_density = 0.3
        else:  # EXPERT
            env_size = (70.0, 40.0)
            num_spots = 12
            obstacle_density = 0.4
        
        # Generate parking spots in rows
        parking_spots = []
        spots_per_row = max(3, num_spots // 2)
        
        for row in range(2):
            y_pos = 10 + row * 15
            for col in range(spots_per_row):
                if len(parking_spots) >= num_spots:
                    break
                    
                x_pos = 8 + col * 6
                if x_pos + 3 < env_size[0]:  # Ensure spot fits
                    spot = ParkingSpot(
                        x=x_pos, y=y_pos,
                        width=2.5, height=5.0,
                        angle=0,  # Perpendicular parking
                        spot_type="perpendicular"
                    )
                    parking_spots.append(spot)
        
        # Generate static obstacles (shopping cart returns, light poles, etc.)
        static_obstacles = []
        num_obstacles = int(env_size[0] * env_size[1] * obstacle_density / 100)
        
        for _ in range(num_obstacles):
            obstacle_type = random.choice(["bollard", "barrier", "vehicle"])
            
            if obstacle_type == "bollard":
                obs = StaticObstacle(
                    x=random.uniform(5, env_size[0] - 5),
                    y=random.uniform(5, env_size[1] - 5),
                    width=0.3, height=0.3,
                    obstacle_type=obstacle_type
                )
            elif obstacle_type == "barrier":
                obs = StaticObstacle(
                    x=random.uniform(5, env_size[0] - 5),
                    y=random.uniform(5, env_size[1] - 5),
                    width=3.0, height=0.2,
                    obstacle_type=obstacle_type
                )
            else:  # vehicle
                obs = StaticObstacle(
                    x=random.uniform(5, env_size[0] - 5),
                    y=random.uniform(5, env_size[1] - 5),
                    width=4.5, height=2.0,
                    obstacle_type=obstacle_type
                )
            
            static_obstacles.append(obs)
        
        return ScenarioConfig(
            name=f"supermarket_{difficulty.value}",
            difficulty=difficulty,
            environment_size=env_size,
            parking_spots=parking_spots,
            static_obstacles=static_obstacles,
            weather=WeatherCondition.CLEAR,
            time_of_day="day",
            dynamic_obstacle_density="light",
            sensor_noise_level=0.1,
            max_episode_steps=200 + int(difficulty.value == "expert") * 100,
            success_tolerance={"position": 0.5, "angle": 0.17}  # 10 degrees
        )
    
    def _generate_street_parking_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate street-side parallel parking scenario."""
        
        env_size = (60.0, 20.0)
        
        # Street parking is naturally more difficult
        if difficulty == ScenarioDifficulty.EASY:
            num_spots = 2
            car_density = 0.3
        else:
            num_spots = 3 + int(difficulty == ScenarioDifficulty.EXPERT)
            car_density = 0.4 + 0.1 * (difficulty == ScenarioDifficulty.HARD or difficulty == ScenarioDifficulty.EXPERT)
        
        # Generate parallel parking spots along the street
        parking_spots = []
        for i in range(num_spots):
            x_pos = 10 + i * 15
            spot = ParkingSpot(
                x=x_pos, y=5,
                width=6.0, height=2.5,  # Parallel parking dimensions
                angle=0,
                spot_type="parallel"
            )
            parking_spots.append(spot)
        
        # Add parked cars as obstacles
        static_obstacles = []
        num_cars = int(env_size[0] * car_density / 6)  # One car per 6 meters on average
        
        for i in range(num_cars):
            x_pos = random.uniform(5, env_size[0] - 5)
            # Place cars along the street
            y_pos = random.choice([3, 17])  # Both sides of street
            
            car = StaticObstacle(
                x=x_pos, y=y_pos,
                width=4.5, height=2.0,
                obstacle_type="vehicle"
            )
            static_obstacles.append(car)
        
        return ScenarioConfig(
            name=f"street_parking_{difficulty.value}",
            difficulty=difficulty,
            environment_size=env_size,
            parking_spots=parking_spots,
            static_obstacles=static_obstacles,
            weather=WeatherCondition.CLEAR,
            time_of_day="day",
            dynamic_obstacle_density="moderate",
            sensor_noise_level=0.15,
            max_episode_steps=300,
            success_tolerance={"position": 0.3, "angle": 0.1}  # Tighter tolerance for parallel parking
        )
    
    def _generate_garage_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate underground garage parking scenario."""
        
        if difficulty == ScenarioDifficulty.EASY:
            env_size = (35.0, 25.0)
            num_spots = 4
        else:
            env_size = (45.0, 30.0)
            num_spots = 6 + int(difficulty == ScenarioDifficulty.EXPERT) * 2
        
        # Garage has tighter spaces and pillars
        parking_spots = []
        for row in range(2):
            for col in range(num_spots // 2):
                x_pos = 8 + col * 5.5  # Tighter spacing
                y_pos = 8 + row * 12
                
                spot = ParkingSpot(
                    x=x_pos, y=y_pos,
                    width=2.3, height=4.8,  # Slightly smaller spots
                    angle=0,
                    spot_type="perpendicular"
                )
                parking_spots.append(spot)
        
        # Add garage pillars and structural elements
        static_obstacles = []
        num_pillars = 3 + int(difficulty.value != "easy")
        
        for i in range(num_pillars):
            pillar = StaticObstacle(
                x=random.uniform(10, env_size[0] - 10),
                y=random.uniform(10, env_size[1] - 10),
                width=1.0, height=1.0,
                obstacle_type="wall"
            )
            static_obstacles.append(pillar)
        
        return ScenarioConfig(
            name=f"garage_{difficulty.value}",
            difficulty=difficulty,
            environment_size=env_size,
            parking_spots=parking_spots,
            static_obstacles=static_obstacles,
            weather=WeatherCondition.CLEAR,
            time_of_day="day",  # Indoor, so weather doesn't matter
            dynamic_obstacle_density="light",
            sensor_noise_level=0.2,  # More sensor noise in confined spaces
            max_episode_steps=250,
            success_tolerance={"position": 0.4, "angle": 0.15}
        )
    
    def _generate_mall_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate shopping mall parking scenario."""
        return self._generate_supermarket_scenario(difficulty)  # Similar layout
    
    def _generate_residential_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate residential area parking scenario."""
        
        env_size = (50.0, 30.0)
        
        # Mix of parallel and perpendicular parking
        parking_spots = []
        
        # Driveway parking (perpendicular)
        for i in range(2):
            spot = ParkingSpot(
                x=15 + i * 20, y=25,
                width=3.0, height=6.0,
                angle=0,
                spot_type="perpendicular"
            )
            parking_spots.append(spot)
        
        # Street parking (parallel)
        for i in range(2):
            spot = ParkingSpot(
                x=20 + i * 15, y=8,
                width=6.0, height=2.5,
                angle=0,
                spot_type="parallel"
            )
            parking_spots.append(spot)
        
        # Add residential obstacles (mailboxes, trees, parked cars)
        static_obstacles = []
        obstacle_types = ["bollard", "vehicle", "barrier"]
        
        for _ in range(3 + int(difficulty.value != "easy")):
            obs_type = random.choice(obstacle_types)
            
            if obs_type == "bollard":  # Mailbox/tree
                obs = StaticObstacle(
                    x=random.uniform(5, env_size[0] - 5),
                    y=random.uniform(5, env_size[1] - 5),
                    width=0.5, height=0.5,
                    obstacle_type=obs_type
                )
            else:
                obs = StaticObstacle(
                    x=random.uniform(5, env_size[0] - 5),
                    y=random.uniform(5, env_size[1] - 5),
                    width=4.5, height=2.0,
                    obstacle_type=obs_type
                )
            
            static_obstacles.append(obs)
        
        return ScenarioConfig(
            name=f"residential_{difficulty.value}",
            difficulty=difficulty,
            environment_size=env_size,
            parking_spots=parking_spots,
            static_obstacles=static_obstacles,
            weather=WeatherCondition.CLEAR,
            time_of_day="day",
            dynamic_obstacle_density="light",
            sensor_noise_level=0.1,
            max_episode_steps=300,
            success_tolerance={"position": 0.5, "angle": 0.17}
        )
    
    def _generate_highway_rest_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate highway rest area parking scenario."""
        
        env_size = (70.0, 40.0)
        
        # Large angled parking spaces
        parking_spots = []
        angle = np.pi / 6  # 30-degree angle
        
        num_spots = 6 + int(difficulty.value != "easy") * 2
        
        for i in range(num_spots):
            x_pos = 15 + (i % 4) * 12
            y_pos = 15 + (i // 4) * 15
            
            spot = ParkingSpot(
                x=x_pos, y=y_pos,
                width=3.0, height=5.5,
                angle=angle,
                spot_type="angled"
            )
            parking_spots.append(spot)
        
        # Add rest area obstacles
        static_obstacles = []
        
        # Picnic tables, trash bins
        for _ in range(4):
            obs = StaticObstacle(
                x=random.uniform(10, env_size[0] - 10),
                y=random.uniform(10, env_size[1] - 10),
                width=2.0, height=1.5,
                obstacle_type="barrier"
            )
            static_obstacles.append(obs)
        
        return ScenarioConfig(
            name=f"highway_rest_{difficulty.value}",
            difficulty=difficulty,
            environment_size=env_size,
            parking_spots=parking_spots,
            static_obstacles=static_obstacles,
            weather=WeatherCondition.CLEAR,
            time_of_day="day",
            dynamic_obstacle_density="moderate",
            sensor_noise_level=0.1,
            max_episode_steps=250,
            success_tolerance={"position": 0.6, "angle": 0.2}  # More tolerant for angled parking
        )
    
    def _generate_airport_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate airport parking scenario."""
        return self._generate_supermarket_scenario(difficulty)  # Similar to supermarket but larger
    
    def _generate_hospital_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate hospital parking scenario."""
        return self._generate_supermarket_scenario(difficulty)
    
    def _generate_school_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate school parking scenario."""
        return self._generate_supermarket_scenario(difficulty)
    
    def _generate_office_scenario(self, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Generate office building parking scenario."""
        return self._generate_garage_scenario(difficulty)  # Similar to garage
    
    def _add_environmental_variations(self, config: ScenarioConfig, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Add weather and lighting variations."""
        
        # Weather effects based on difficulty
        if difficulty == ScenarioDifficulty.EASY:
            weather_options = [WeatherCondition.CLEAR]
        elif difficulty == ScenarioDifficulty.MEDIUM:
            weather_options = [WeatherCondition.CLEAR, WeatherCondition.LIGHT_RAIN]
        elif difficulty == ScenarioDifficulty.HARD:
            weather_options = [WeatherCondition.CLEAR, WeatherCondition.LIGHT_RAIN, WeatherCondition.FOG]
        else:  # EXPERT
            weather_options = list(WeatherCondition)
        
        config.weather = random.choice(weather_options)
        
        # Time of day variations
        if difficulty in [ScenarioDifficulty.HARD, ScenarioDifficulty.EXPERT]:
            time_options = ["day", "night", "dawn", "dusk"]
        else:
            time_options = ["day", "dawn", "dusk"]
        
        config.time_of_day = random.choice(time_options)
        
        # Adjust sensor noise based on conditions
        if config.weather in [WeatherCondition.HEAVY_RAIN, WeatherCondition.FOG, WeatherCondition.SNOW]:
            config.sensor_noise_level *= 2.0
        elif config.weather == WeatherCondition.LIGHT_RAIN:
            config.sensor_noise_level *= 1.5
        
        if config.time_of_day in ["night", "dawn", "dusk"]:
            config.sensor_noise_level *= 1.3
        
        return config
    
    def _add_dynamic_elements(self, config: ScenarioConfig, difficulty: ScenarioDifficulty) -> ScenarioConfig:
        """Add dynamic obstacle density based on difficulty."""
        
        if difficulty == ScenarioDifficulty.EASY:
            config.dynamic_obstacle_density = "none"
        elif difficulty == ScenarioDifficulty.MEDIUM:
            config.dynamic_obstacle_density = "light"
        elif difficulty == ScenarioDifficulty.HARD:
            config.dynamic_obstacle_density = "moderate"
        else:  # EXPERT
            config.dynamic_obstacle_density = "heavy"
        
        return config
    
    def _load_scenario_templates(self) -> Dict[str, Any]:
        """Load predefined scenario templates."""
        # This could be loaded from configuration files in a real implementation
        return {}
    
    def generate_training_curriculum(self, total_scenarios: int = 1000) -> List[ScenarioConfig]:
        """
        Generate a curriculum of scenarios for progressive training.
        
        Args:
            total_scenarios: Total number of scenarios to generate
            
        Returns:
            List of scenarios ordered by difficulty
        """
        scenarios = []
        
        # Distribution of difficulties
        difficulty_distribution = {
            ScenarioDifficulty.EASY: 0.4,
            ScenarioDifficulty.MEDIUM: 0.3,
            ScenarioDifficulty.HARD: 0.2,
            ScenarioDifficulty.EXPERT: 0.1
        }
        
        scenario_types = [
            "supermarket", "street_parking", "garage", "mall", "residential",
            "highway_rest", "airport", "hospital", "school", "office"
        ]
        
        for i in range(total_scenarios):
            # Progressive difficulty - start easier, get harder
            progress = i / total_scenarios
            
            if progress < 0.4:
                difficulty = ScenarioDifficulty.EASY
            elif progress < 0.7:
                difficulty = ScenarioDifficulty.MEDIUM
            elif progress < 0.9:
                difficulty = ScenarioDifficulty.HARD
            else:
                difficulty = ScenarioDifficulty.EXPERT
            
            # Add some randomness
            if random.random() < 0.2:  # 20% chance to change difficulty
                difficulty = random.choice(list(ScenarioDifficulty))
            
            scenario_type = random.choice(scenario_types)
            scenario = self.generate_scenario(difficulty, scenario_type)
            scenarios.append(scenario)
        
        return scenarios


# Predefined scenario collections for specific training purposes
SCENARIO_COLLECTIONS = {
    "basic_training": [
        ("supermarket", ScenarioDifficulty.EASY),
        ("garage", ScenarioDifficulty.EASY),
        ("residential", ScenarioDifficulty.MEDIUM)
    ],
    
    "intermediate_training": [
        ("street_parking", ScenarioDifficulty.MEDIUM),
        ("mall", ScenarioDifficulty.MEDIUM),
        ("garage", ScenarioDifficulty.HARD)
    ],
    
    "advanced_training": [
        ("street_parking", ScenarioDifficulty.HARD),
        ("highway_rest", ScenarioDifficulty.HARD),
        ("airport", ScenarioDifficulty.EXPERT)
    ],
    
    "mixed_scenarios": [
        ("supermarket", ScenarioDifficulty.EASY),
        ("street_parking", ScenarioDifficulty.MEDIUM),
        ("garage", ScenarioDifficulty.HARD),
        ("residential", ScenarioDifficulty.EXPERT)
    ]
} 