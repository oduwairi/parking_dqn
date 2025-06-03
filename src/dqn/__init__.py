"""
DQN Package Initialization

Phase 4 Components:
- DQN Network Architecture (main and target networks)
- Experience Replay Buffer (standard and prioritized)
- DQN Agent (complete training agent)
- Loss Functions (Huber loss and training utilities)
"""

# Network components
from .network import DQNNetwork, DoubleDQNNetwork, create_dqn_networks

# Replay buffer components  
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Experience, create_replay_buffer

# Loss function components
from .loss_functions import (
    HuberLoss, DQNLoss, GradientManager, 
    LearningRateScheduler, EpsilonScheduler, 
    calculate_training_metrics
)

# Main agent
from .agent import DQNAgent, create_dqn_agent

__all__ = [
    # Networks
    'DQNNetwork',
    'DoubleDQNNetwork', 
    'create_dqn_networks',
    
    # Replay buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'Experience',
    'create_replay_buffer',
    
    # Loss functions and utilities
    'HuberLoss',
    'DQNLoss',
    'GradientManager',
    'LearningRateScheduler',
    'EpsilonScheduler',
    'calculate_training_metrics',
    
    # Main agent
    'DQNAgent',
    'create_dqn_agent'
] 