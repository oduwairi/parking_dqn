#!/usr/bin/env python3
"""
DQN Agent

Main DQN agent that integrates all components:
- Deep Q-Network (main and target)
- Experience replay buffer
- Epsilon-greedy exploration policy
- Training procedures with paper specifications

Author: Based on "Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import random
import time
import logging

from .network import DQNNetwork, DoubleDQNNetwork, create_dqn_networks
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, create_replay_buffer
from .loss_functions import DQNLoss, GradientManager, LearningRateScheduler, EpsilonScheduler


class DQNAgent:
    """
    Complete DQN Agent for autonomous parking.
    
    Implements the DQN algorithm with:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Double DQN (optional)
    - Prioritized experience replay (optional)
    """
    
    def __init__(self,
                 state_dim: int = 12,
                 action_dim: int = 7,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 replay_buffer_size: int = 100000,
                 target_update_freq: int = 1000,
                 soft_update_tau: float = 1e-3,
                 use_double_dqn: bool = True,
                 use_prioritized_replay: bool = False,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """
        Initialize DQN agent with paper specifications.
        
        Args:
            state_dim: State space dimension (12 as per paper)
            action_dim: Action space dimension (7 as per paper)
            learning_rate: Learning rate α (1e-3 as per paper)
            gamma: Discount factor γ (0.9-0.95 as per paper)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Training batch size (64 as per paper)
            replay_buffer_size: Replay buffer capacity (~10^5 as per paper)
            target_update_freq: Target network update frequency (1000 as per paper)
            soft_update_tau: Soft update rate τ (1e-3 as per paper)
            use_double_dqn: Whether to use Double DQN
            use_prioritized_replay: Whether to use prioritized experience replay
            device: Computing device
            seed: Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.soft_update_tau = soft_update_tau
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Set random seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Initialize networks
        self.main_network, self.target_network = create_dqn_networks(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            use_double_dqn=use_double_dqn
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        buffer_type = "prioritized" if use_prioritized_replay else "standard"
        self.replay_buffer = create_replay_buffer(
            buffer_type=buffer_type,
            capacity=replay_buffer_size,
            state_dim=state_dim,
            seed=seed
        )
        
        # Initialize loss function
        self.loss_function = DQNLoss(gamma=gamma)
        
        # Initialize schedulers
        self.epsilon_scheduler = EpsilonScheduler(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay
        )
        
        self.lr_scheduler = LearningRateScheduler(
            initial_lr=learning_rate
        )
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.training_history = {
            'losses': [],
            'rewards': [],
            'epsilons': [],
            'q_values': [],
            'learning_rates': []
        }
        
        # Performance tracking
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.success_rate = 0.0
        
        print(f"✅ DQN Agent initialized:")
        print(f"   - Device: {self.device}")
        print(f"   - Network type: {'Double DQN' if use_double_dqn else 'Standard DQN'}")
        print(f"   - Replay buffer: {'Prioritized' if use_prioritized_replay else 'Standard'}")
        print(f"   - Network parameters: {self.main_network.get_network_info()['total_parameters']:,}")
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, use scheduler)
            
        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon_scheduler.current_epsilon
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action (exploration)
            action = np.random.randint(0, self.action_dim)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.main_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def can_train(self) -> bool:
        """Check if agent has enough experiences to start training."""
        return self.replay_buffer.can_provide_sample(self.batch_size)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics
        """
        if not self.can_train():
            return {'error': 'Not enough experiences for training'}
        
        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = None
            indices = None
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute loss
        if self.use_double_dqn:
            loss, metrics = self.loss_function.compute_double_dqn_loss(
                self.main_network, self.target_network,
                states, actions, rewards, next_states, dones, weights
            )
        else:
            loss, metrics = self.loss_function.compute_dqn_loss(
                self.main_network, self.target_network,
                states, actions, rewards, next_states, dones, weights
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        grad_norm = GradientManager.clip_gradients(self.main_network, max_norm=10.0)
        
        # Update weights
        self.optimizer.step()
        
        # Update prioritized replay buffer
        if self.use_prioritized_replay and indices is not None:
            with torch.no_grad():
                current_q = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                if self.use_double_dqn:
                    next_q_main = self.main_network(next_states)
                    next_actions = next_q_main.argmax(1, keepdim=True)
                    next_q_target = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_q_target = self.target_network(next_states).max(1)[0]
                
                target_q = rewards + (self.gamma * next_q_target * (1 - dones.float()))
                td_errors = torch.abs(current_q - target_q).cpu().numpy()
                
                self.replay_buffer.update_priorities(indices, td_errors)
                self.replay_buffer.step_frame()
        
        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.main_network)
        
        # Soft update (alternative to hard update)
        # self.target_network.soft_update_from(self.main_network, self.soft_update_tau)
        
        # Update learning rate
        current_lr = self.lr_scheduler.update_optimizer(self.optimizer, self.training_step)
        
        # Update epsilon
        current_epsilon = self.epsilon_scheduler.step()
        
        # Update training step
        self.training_step += 1
        
        # Compile metrics
        training_metrics = {
            'loss': metrics['loss'],
            'mean_q_value': metrics['mean_q_value'],
            'mean_target_q': metrics['mean_target_q'],
            'td_error': metrics['mean_td_error'],
            'gradient_norm': grad_norm,
            'learning_rate': current_lr,
            'epsilon': current_epsilon,
            'training_step': self.training_step,
            'buffer_size': self.replay_buffer.size
        }
        
        # Store in history
        self.training_history['losses'].append(metrics['loss'])
        self.training_history['q_values'].append(metrics['mean_q_value'])
        self.training_history['epsilons'].append(current_epsilon)
        self.training_history['learning_rates'].append(current_lr)
        
        return training_metrics
    
    def update_episode_stats(self, episode_reward: float, success: bool):
        """Update episode statistics."""
        self.episode_count += 1
        self.total_reward += episode_reward
        self.recent_rewards.append(episode_reward)
        
        # Keep only recent rewards (for moving average)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # Update best reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        # Store in history
        self.training_history['rewards'].append(episode_reward)
        
        # Update success rate (moving average)
        recent_successes = sum(1 for r in self.recent_rewards[-20:] if r > 50)  # Threshold for success
        self.success_rate = recent_successes / min(len(self.recent_rewards), 20)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        buffer_info = self.replay_buffer.get_buffer_info()
        network_info = self.main_network.get_network_info()
        
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'current_epsilon': self.epsilon_scheduler.current_epsilon,
            'success_rate': self.success_rate,
            'best_reward': self.best_reward,
            'average_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'buffer_usage': buffer_info['usage_percent'],
            'network_parameters': network_info['total_parameters'],
            'device': str(self.device),
            'use_double_dqn': self.use_double_dqn,
            'use_prioritized_replay': self.use_prioritized_replay
        }
    
    def save_model(self, filepath: str):
        """Save agent model and training state."""
        checkpoint = {
            'main_network_state': self.main_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'training_history': self.training_history,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'use_double_dqn': self.use_double_dqn,
                'use_prioritized_replay': self.use_prioritized_replay
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load agent model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.main_network.load_state_dict(checkpoint['main_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.training_history = checkpoint['training_history']
        
        print(f"✅ Model loaded from {filepath}")
        print(f"   - Training step: {self.training_step}")
        print(f"   - Episodes: {self.episode_count}")
    
    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.main_network.eval()
        self.target_network.eval()
    
    def set_train_mode(self):
        """Set networks to training mode."""
        self.main_network.train()
        # Target network stays in eval mode
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.main_network(state_tensor)
            return q_values.cpu().numpy().flatten()
    
    def reset_training_history(self):
        """Reset training history (for fresh start)."""
        self.training_history = {
            'losses': [],
            'rewards': [],
            'epsilons': [],
            'q_values': [],
            'learning_rates': []
        }
        self.recent_rewards = []
        self.training_step = 0
        self.episode_count = 0


def create_dqn_agent(config: Dict[str, Any]) -> DQNAgent:
    """
    Factory function to create DQN agent from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DQN agent
    """
    return DQNAgent(**config)


if __name__ == "__main__":
    # Test DQN agent creation and basic functionality
    print("Testing DQN Agent...")
    
    # Create agent with default parameters
    agent = DQNAgent(seed=42)
    
    # Test action selection
    test_state = np.random.randn(12)
    action = agent.select_action(test_state)
    print(f"✅ Action selected: {action}")
    
    # Test experience storage
    next_state = np.random.randn(12)
    agent.store_experience(test_state, action, 1.0, next_state, False)
    print(f"✅ Experience stored")
    
    # Test agent info
    info = agent.get_agent_info()
    print(f"✅ Agent info: {info}")
    
    print("DQN Agent test completed successfully!") 