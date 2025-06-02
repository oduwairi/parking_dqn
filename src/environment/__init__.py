"""
Environment Package Initialization
Contains all environment-related modules for the autonomous parking simulation.

Phase 1 Components:
- ParkingEnv: Main OpenAI Gym environment
- CarAgent: Car physics and kinematic model
- ParkingRenderer: Visualization and rendering

Phase 2 Components:
- ActionSpace: 7 discrete actions from research paper
- SensorArray: 8-directional distance sensors
- RewardFunction: Comprehensive reward system

Phase 3 Components:
- ObstacleManager: Static obstacle management
- ParkingSpotManager: Parking validation and tolerances
- CollisionDetector: Collision detection system
"""

# Core environment components
from .parking_env import ParkingEnv
from .car_agent import CarAgent
from .renderer import ParkingRenderer

# Phase 2 components
from .action_space import ActionSpace, ActionType
from .sensors import SensorArray, DistanceSensor
from .rewards import RewardFunction, RewardType

# Phase 3 components
from .obstacles import ObstacleManager, Obstacle, ObstacleType
from .parking_spots import ParkingSpotManager, ParkingSpot, ParkingSpotType
from .collision_detection import CollisionDetector, CollisionInfo, CollisionType

__all__ = [
    # Core environment
    'ParkingEnv',
    'CarAgent', 
    'ParkingRenderer',
    
    # Phase 2: Action space & sensors
    'ActionSpace',
    'ActionType',
    'SensorArray',
    'DistanceSensor',
    'RewardFunction',
    'RewardType',
    
    # Phase 3: Obstacles & collision detection
    'ObstacleManager',
    'Obstacle',
    'ObstacleType',
    'ParkingSpotManager',
    'ParkingSpot',
    'ParkingSpotType',
    'CollisionDetector',
    'CollisionInfo',
    'CollisionType'
] 