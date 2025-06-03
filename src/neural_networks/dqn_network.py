import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for autonomous parking with proven architecture.
    
    Based on successful research implementations:
    - He initialization for ReLU activations
    - Proper layer sizing for parking state space
    - Gradient-friendly architecture
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = None):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Default proven architecture for parking DQN
        if hidden_sizes is None:
            hidden_sizes = [512, 512, 256]  # Proven for parking tasks
        
        # Create network layers
        layers = []
        input_size = state_size
        
        # Hidden layers with ReLU activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Apply He initialization (proven critical for ReLU networks)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Apply He initialization - proven critical for DQN convergence.
        
        Research shows this is essential for ReLU networks to converge properly.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # He initialization with scale=2 (proven in research)
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                # Initialize biases to small positive values (proven technique)
                nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture - proven to improve parking performance.
    
    Separates state-value and advantage estimation for better learning.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = None):
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Default proven architecture
        if hidden_sizes is None:
            hidden_sizes = [512, 512]
        
        # Shared feature extraction layers
        feature_layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            feature_layers.append(nn.Linear(input_size, hidden_size))
            feature_layers.append(nn.ReLU())
            input_size = hidden_size
        
        self.feature_layers = nn.Sequential(*feature_layers)
        
        # Dueling streams
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream: estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
        # Apply He initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply He initialization to all layers."""
        for module in [self.feature_layers, self.value_stream, self.advantage_stream]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state):
        """
        Forward pass through dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        """
        features = self.feature_layers(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams
        # Subtract mean advantage for identifiability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values 