#!/usr/bin/env python3
"""
Loss Functions and Training Utilities

Implements loss functions and training utilities for DQN:
- Huber loss function (robust to outliers)
- Training metrics calculation
- Gradient clipping utilities
- Learning rate scheduling

Author: Based on "Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import math


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss) for DQN training.
    
    Combines the benefits of L1 and L2 loss:
    - Quadratic for small errors (stable gradients)
    - Linear for large errors (robust to outliers)
    
    Loss = 0.5 * (error^2) if |error| <= δ
    Loss = δ * (|error| - 0.5 * δ) if |error| > δ
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold parameter (default: 1.0 as commonly used in DQN)
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Huber loss.
        
        Args:
            predictions: Predicted Q-values
            targets: Target Q-values
            
        Returns:
            Huber loss tensor
        """
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Quadratic for small errors, linear for large errors
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        
        return loss.mean()


class DQNLoss:
    """
    Complete DQN loss calculation with support for:
    - Standard DQN
    - Double DQN
    - Prioritized experience replay
    """
    
    def __init__(self, 
                 loss_function: str = "huber",
                 delta: float = 1.0,
                 gamma: float = 0.95):
        """
        Initialize DQN loss calculator.
        
        Args:
            loss_function: "huber", "mse", or "smooth_l1"
            delta: Huber loss delta parameter
            gamma: Discount factor (0.95 as per paper)
        """
        self.gamma = gamma
        
        # Select loss function
        if loss_function.lower() == "huber":
            self.criterion = HuberLoss(delta=delta)
        elif loss_function.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif loss_function.lower() == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
    
    def compute_dqn_loss(self,
                         q_network: nn.Module,
                         target_network: nn.Module,
                         states: torch.Tensor,
                         actions: torch.Tensor,
                         rewards: torch.Tensor,
                         next_states: torch.Tensor,
                         dones: torch.Tensor,
                         weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute standard DQN loss.
        
        Args:
            q_network: Main Q-network
            target_network: Target Q-network
            states: Current states batch
            actions: Actions taken batch
            rewards: Rewards received batch
            next_states: Next states batch
            dones: Done flags batch
            weights: Importance sampling weights (for prioritized replay)
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = states.size(0)
        
        # Get current Q-values
        current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get target Q-values
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones.float()))
        
        # Calculate loss
        if weights is not None:
            # Weighted loss for prioritized experience replay
            loss = (weights * self.criterion(current_q_values, target_q_values)).mean()
        else:
            loss = self.criterion(current_q_values, target_q_values)
        
        # Calculate metrics
        with torch.no_grad():
            td_errors = torch.abs(current_q_values - target_q_values)
            metrics = {
                'loss': loss.item(),
                'mean_q_value': current_q_values.mean().item(),
                'mean_target_q': target_q_values.mean().item(),
                'mean_td_error': td_errors.mean().item(),
                'max_td_error': td_errors.max().item(),
                'batch_size': batch_size
            }
        
        return loss, metrics
    
    def compute_double_dqn_loss(self,
                               q_network: nn.Module,
                               target_network: nn.Module,
                               states: torch.Tensor,
                               actions: torch.Tensor,
                               rewards: torch.Tensor,
                               next_states: torch.Tensor,
                               dones: torch.Tensor,
                               weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Double DQN loss to reduce overestimation bias.
        
        Uses main network for action selection and target network for value estimation.
        """
        batch_size = states.size(0)
        
        # Get current Q-values
        current_q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            # Use main network to select best actions
            next_q_values_main = q_network(next_states)
            next_actions = next_q_values_main.argmax(1, keepdim=True)
            
            # Use target network to evaluate selected actions
            next_q_values_target = target_network(next_states).gather(1, next_actions).squeeze(1)
            
            # Compute targets
            target_q_values = rewards + (self.gamma * next_q_values_target * (1 - dones.float()))
        
        # Calculate loss
        if weights is not None:
            loss = (weights * self.criterion(current_q_values, target_q_values)).mean()
        else:
            loss = self.criterion(current_q_values, target_q_values)
        
        # Calculate metrics
        with torch.no_grad():
            td_errors = torch.abs(current_q_values - target_q_values)
            metrics = {
                'loss': loss.item(),
                'mean_q_value': current_q_values.mean().item(),
                'mean_target_q': target_q_values.mean().item(),
                'mean_td_error': td_errors.mean().item(),
                'max_td_error': td_errors.max().item(),
                'batch_size': batch_size,
                'double_dqn_used': True
            }
        
        return loss, metrics


class GradientManager:
    """Utilities for gradient management during training."""
    
    @staticmethod
    def clip_gradients(model: nn.Module, max_norm: float = 10.0) -> float:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            model: Neural network model
            max_norm: Maximum gradient norm
            
        Returns:
            Gradient norm before clipping
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return grad_norm.item()
    
    @staticmethod
    def get_gradient_stats(model: nn.Module) -> Dict[str, float]:
        """Get gradient statistics for monitoring."""
        total_norm = 0.0
        param_count = 0
        max_grad = 0.0
        min_grad = float('inf')
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += param.numel()
                
                max_grad = max(max_grad, param.grad.data.abs().max().item())
                min_grad = min(min_grad, param.grad.data.abs().min().item())
        
        total_norm = total_norm ** (1. / 2)
        
        return {
            'gradient_norm': total_norm,
            'max_gradient': max_grad,
            'min_gradient': min_grad if min_grad != float('inf') else 0.0,
            'param_count': param_count
        }


class LearningRateScheduler:
    """Learning rate scheduling for DQN training."""
    
    def __init__(self, 
                 initial_lr: float = 1e-3,
                 schedule_type: str = "exponential",
                 decay_rate: float = 0.95,
                 decay_steps: int = 10000,
                 min_lr: float = 1e-6):
        """
        Initialize learning rate scheduler.
        
        Args:
            initial_lr: Initial learning rate (1e-3 as per paper)
            schedule_type: "exponential", "step", or "cosine"
            decay_rate: Decay rate for exponential/step decay
            decay_steps: Steps between decay for step scheduling
            min_lr: Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        if self.schedule_type == "exponential":
            lr = self.initial_lr * (self.decay_rate ** (step / self.decay_steps))
        elif self.schedule_type == "step":
            lr = self.initial_lr * (self.decay_rate ** (step // self.decay_steps))
        elif self.schedule_type == "cosine":
            lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * step / self.decay_steps)) / 2
        else:
            lr = self.initial_lr  # Constant learning rate
        
        return max(lr, self.min_lr)
    
    def update_optimizer(self, optimizer: torch.optim.Optimizer, step: int):
        """Update optimizer learning rate."""
        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EpsilonScheduler:
    """Epsilon-greedy exploration scheduling."""
    
    def __init__(self,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 decay_type: str = "exponential"):
        """
        Initialize epsilon scheduler.
        
        Args:
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate
            decay_type: "exponential" or "linear"
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decay_type = decay_type
        self.current_epsilon = epsilon_start
    
    def get_epsilon(self, step: int) -> float:
        """Get epsilon for current step."""
        if self.decay_type == "exponential":
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     math.exp(-step * self.epsilon_decay)
        else:  # linear
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                     min(step * self.epsilon_decay, 1.0)
        
        self.current_epsilon = max(epsilon, self.epsilon_end)
        return self.current_epsilon
    
    def step(self):
        """Step the epsilon scheduler (for exponential decay)."""
        if self.decay_type == "exponential":
            self.current_epsilon = max(
                self.current_epsilon * self.epsilon_decay,
                self.epsilon_end
            )
        return self.current_epsilon


def calculate_training_metrics(losses: List[float], 
                             rewards: List[float],
                             epsilons: List[float],
                             window_size: int = 100) -> Dict[str, float]:
    """Calculate comprehensive training metrics."""
    if not losses:
        return {}
    
    recent_losses = losses[-window_size:]
    recent_rewards = rewards[-window_size:]
    recent_epsilons = epsilons[-window_size:]
    
    metrics = {
        'mean_loss': np.mean(recent_losses),
        'std_loss': np.std(recent_losses),
        'mean_reward': np.mean(recent_rewards),
        'std_reward': np.std(recent_rewards),
        'current_epsilon': recent_epsilons[-1] if recent_epsilons else 0.0,
        'training_steps': len(losses)
    }
    
    return metrics


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Test Huber loss
    huber_loss = HuberLoss()
    predictions = torch.randn(32, 1)
    targets = torch.randn(32, 1)
    loss = huber_loss(predictions, targets)
    print(f"✅ Huber loss computed: {loss.item():.4f}")
    
    # Test epsilon scheduler
    epsilon_scheduler = EpsilonScheduler()
    epsilons = [epsilon_scheduler.get_epsilon(i) for i in range(100)]
    print(f"✅ Epsilon decay: {epsilons[0]:.3f} -> {epsilons[-1]:.3f}")
    
    # Test learning rate scheduler
    lr_scheduler = LearningRateScheduler()
    lrs = [lr_scheduler.get_lr(i) for i in range(100)]
    print(f"✅ LR decay: {lrs[0]:.6f} -> {lrs[-1]:.6f}")
    
    print("Loss Functions test completed successfully!") 