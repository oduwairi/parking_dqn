#!/usr/bin/env python3
"""
DQN Network Architecture

Implements the Deep Q-Network architecture as specified in the research paper:
- Input: 12-dimensional state vector [x, y, θ, v, d_1...d_8]
- Hidden layers: 3 layers × 256 neurons each using ReLU activation,
- Output: Q-values for 7 discrete actions
- Support for both main network (θ) and target network (θ⁻)

Author: Based on "Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class DQNNetwork(nn.Module):
    """
    Deep Q-Network implementation following paper specifications.
    
    Architecture:
    - Input layer: 12 neurons (state vector)
    - Hidden layer 1: 256 neurons + ReLU
    - Hidden layer 2: 256 neurons + ReLU  
    - Hidden layer 3: 256 neurons + ReLU
    - Output layer: 7 neurons (Q-values for actions)
    """
    
    def __init__(self, 
                 state_dim: int = 12,
                 action_dim: int = 7,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 3,
                 dropout_rate: float = 0.1):
        """
        Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space (default: 12)
            action_dim: Number of discrete actions (default: 7)
            hidden_dim: Number of neurons per hidden layer (default: 256)
            num_hidden_layers: Number of hidden layers (default: 3)
            dropout_rate: Dropout rate for regularization (default: 0.1)
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers (3 layers × 256 neurons as per paper)
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        # Create sequential network
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        # Ensure input is the correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        q_values = self.network(state)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> Tuple[int, torch.Tensor]:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            epsilon: Exploration probability
            
        Returns:
            Tuple of (action_index, q_values)
        """
        if np.random.random() < epsilon:
            # Random action (exploration)
            action = np.random.randint(0, self.action_dim)
            q_values = self.forward(state)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax().item()
        
        return action, q_values
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get information about the network architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'num_hidden_layers': self.num_hidden_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'architecture_summary': str(self.network)
        }
    
    def copy_weights_from(self, source_network: 'DQNNetwork'):
        """Copy weights from another DQN network (for target network updates)."""
        self.load_state_dict(source_network.state_dict())
    
    def soft_update_from(self, source_network: 'DQNNetwork', tau: float = 0.001):
        """
        Soft update weights from source network using polyak averaging.
        
        θ_target = τ * θ_main + (1 - τ) * θ_target
        
        Args:
            source_network: Source network to copy from
            tau: Soft update rate (default: 0.001 as per paper)
        """
        for target_param, source_param in zip(self.parameters(), source_network.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


class DoubleDQNNetwork(DQNNetwork):
    """
    Double DQN variant that reduces overestimation bias.
    Uses main network for action selection and target network for value estimation.
    """
    
    def __init__(self, *args, **kwargs):
        super(DoubleDQNNetwork, self).__init__(*args, **kwargs)
        self.network_type = "DoubleDQN"
    
    def compute_double_dqn_target(self, 
                                  next_states: torch.Tensor,
                                  rewards: torch.Tensor,
                                  dones: torch.Tensor,
                                  target_network: 'DQNNetwork',
                                  gamma: float = 0.95) -> torch.Tensor:
        """
        Compute Double DQN target values.
        
        Args:
            next_states: Next state batch
            rewards: Reward batch
            dones: Done flags batch
            target_network: Target network for value estimation
            gamma: Discount factor
            
        Returns:
            Target Q-values
        """
        with torch.no_grad():
            # Use main network to select actions
            next_q_values_main = self.forward(next_states)
            next_actions = next_q_values_main.argmax(dim=1, keepdim=True)
            
            # Use target network to evaluate selected actions
            next_q_values_target = target_network.forward(next_states)
            next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)
            
            # Compute target: r + γ * Q_target(s', argmax_a Q_main(s', a)) * (1 - done)
            targets = rewards + (gamma * next_q_values * (1 - dones.float()))
        
        return targets


def create_dqn_networks(state_dim: int = 12, 
                       action_dim: int = 7,
                       device: Optional[torch.device] = None,
                       use_double_dqn: bool = True) -> Tuple[DQNNetwork, DQNNetwork]:
    """
    Create main and target DQN networks.
    
    Args:
        state_dim: State space dimension
        action_dim: Action space dimension  
        device: Torch device to use
        use_double_dqn: Whether to use Double DQN variant
        
    Returns:
        Tuple of (main_network, target_network)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose network type
    NetworkClass = DoubleDQNNetwork if use_double_dqn else DQNNetwork
    
    # Create networks
    main_network = NetworkClass(state_dim=state_dim, action_dim=action_dim)
    target_network = NetworkClass(state_dim=state_dim, action_dim=action_dim)
    
    # Move to device
    main_network.to(device)
    target_network.to(device)
    
    # Initialize target network with same weights as main network
    target_network.copy_weights_from(main_network)
    
    # Set target network to evaluation mode
    target_network.eval()
    
    return main_network, target_network


if __name__ == "__main__":
    # Quick test of network architecture
    print("Testing DQN Network Architecture...")
    
    # Create networks
    main_net, target_net = create_dqn_networks()
    
    # Test forward pass
    test_state = torch.randn(1, 12)  # Batch size 1, state dim 12
    q_values = main_net(test_state)
    
    print(f"✅ Network created successfully")
    print(f"✅ Input shape: {test_state.shape}")
    print(f"✅ Output shape: {q_values.shape}")
    print(f"✅ Network info: {main_net.get_network_info()}")
    
    # Test action selection
    action, q_vals = main_net.get_action(test_state, epsilon=0.1)
    print(f"✅ Action selection working: action={action}")
    
    print("DQN Network test completed successfully!") 