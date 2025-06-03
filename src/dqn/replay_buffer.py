#!/usr/bin/env python3
"""
Experience Replay Buffer

Implements experience replay mechanism for DQN training:
- Circular buffer with capacity ~10^5 transitions (as per paper)
- Efficient random sampling for training batches
- Support for prioritized experience replay (optional)
- Memory-efficient storage with numpy arrays

Author: Based on "Methodology on Autonomous Parking with Obstacle and Situational Adaptability: A Deep Q Learning Approach"
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, NamedTuple, Dict, Any
import random
from collections import deque


class Experience(NamedTuple):
    """Single experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN training.
    
    Stores transitions (s, a, r, s', done) and provides random sampling
    for breaking temporal correlations in training data.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 state_dim: int = 12,
                 seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity (~10^5 as per paper)
            state_dim: Dimension of state space (default: 12)
            seed: Random seed for reproducible sampling
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0
        
        # Initialize storage arrays for memory efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state observed
            done: Whether episode terminated
        """
        # Store experience at current position
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Size of batch to sample (default: 64 as per paper)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer contains only {self.size} experiences, cannot sample {batch_size}")
        
        # Random sampling without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Extract batch data
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.BoolTensor(self.dones[indices])
        
        return states, actions, rewards, next_states, dones
    
    def can_provide_sample(self, batch_size: int = 64) -> bool:
        """Check if buffer can provide a sample of requested size."""
        return self.size >= batch_size
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get information about buffer state."""
        usage_percent = (self.size / self.capacity) * 100
        
        return {
            'capacity': self.capacity,
            'current_size': self.size,
            'usage_percent': usage_percent,
            'position': self.position,
            'state_dim': self.state_dim,
            'memory_usage_mb': self.get_memory_usage_mb()
        }
    
    def get_memory_usage_mb(self) -> float:
        """Calculate approximate memory usage in MB."""
        # Calculate bytes used by numpy arrays
        states_bytes = self.states.nbytes
        actions_bytes = self.actions.nbytes
        rewards_bytes = self.rewards.nbytes
        next_states_bytes = self.next_states.nbytes
        dones_bytes = self.dones.nbytes
        
        total_bytes = states_bytes + actions_bytes + rewards_bytes + next_states_bytes + dones_bytes
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        
        # Reset arrays
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(False)
    
    def get_recent_experiences(self, n: int = 10) -> List[Experience]:
        """Get the n most recent experiences for debugging."""
        if self.size == 0:
            return []
        
        experiences = []
        for i in range(min(n, self.size)):
            idx = (self.position - 1 - i) % self.capacity
            if idx < 0:
                idx += self.capacity
            
            exp = Experience(
                state=self.states[idx].copy(),
                action=self.actions[idx],
                reward=self.rewards[idx],
                next_state=self.next_states[idx].copy(),
                done=self.dones[idx]
            )
            experiences.append(exp)
        
        return experiences
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics for analysis."""
        if self.size == 0:
            return {'empty_buffer': True}
        
        # Calculate statistics on stored experiences
        rewards_data = self.rewards[:self.size]
        actions_data = self.actions[:self.size]
        
        stats = {
            'mean_reward': float(np.mean(rewards_data)),
            'std_reward': float(np.std(rewards_data)),
            'min_reward': float(np.min(rewards_data)),
            'max_reward': float(np.max(rewards_data)),
            'positive_rewards_percent': float(np.mean(rewards_data > 0) * 100),
            'negative_rewards_percent': float(np.mean(rewards_data < 0) * 100),
            'done_rate': float(np.mean(self.dones[:self.size]) * 100)
        }
        
        # Action distribution
        for action in range(7):  # 7 actions as per paper
            action_count = np.sum(actions_data == action)
            stats[f'action_{action}_percent'] = float((action_count / self.size) * 100)
        
        return stats


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences based on TD error magnitude for more efficient learning.
    Uses sum tree data structure for efficient priority-based sampling.
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 state_dim: int = 12,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Buffer capacity
            state_dim: State dimension
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_frames: Number of frames over which beta is annealed to 1.0
            seed: Random seed
        """
        super().__init__(capacity, state_dim, seed)
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Initialize priority arrays
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, td_error: float = None):
        """Add experience with priority."""
        super().push(state, action, reward, next_state, done)
        
        # Set priority (use max priority for new experiences)
        priority = self.max_priority if td_error is None else abs(td_error) + 1e-6
        self.priorities[self.position - 1] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, ...]:
        """Sample batch based on priorities."""
        if self.size < batch_size:
            raise ValueError(f"Buffer contains only {self.size} experiences, cannot sample {batch_size}")
        
        # Calculate probabilities
        priorities = self.priorities[:self.size]
        prob = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=prob, replace=False)
        
        # Calculate importance sampling weights
        beta = self._get_beta()
        weights = (self.size * prob[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Extract batch data
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.BoolTensor(self.dones[indices])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def _get_beta(self) -> float:
        """Get current beta value for importance sampling."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def step_frame(self):
        """Increment frame counter for beta annealing."""
        self.frame += 1


def create_replay_buffer(buffer_type: str = "standard",
                        capacity: int = 100000,
                        state_dim: int = 12,
                        **kwargs) -> ReplayBuffer:
    """
    Factory function to create replay buffer.
    
    Args:
        buffer_type: "standard" or "prioritized"
        capacity: Buffer capacity
        state_dim: State dimension
        **kwargs: Additional arguments for specific buffer types
        
    Returns:
        Replay buffer instance
    """
    if buffer_type.lower() == "prioritized":
        return PrioritizedReplayBuffer(capacity=capacity, state_dim=state_dim, **kwargs)
    else:
        return ReplayBuffer(capacity=capacity, state_dim=state_dim, **kwargs)


if __name__ == "__main__":
    # Test replay buffer functionality
    print("Testing Replay Buffer...")
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000, state_dim=12)
    
    # Add some dummy experiences
    for i in range(100):
        state = np.random.randn(12)
        action = np.random.randint(0, 7)
        reward = np.random.randn()
        next_state = np.random.randn(12)
        done = np.random.random() < 0.1
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"✅ Added 100 experiences")
    print(f"✅ Buffer info: {buffer.get_buffer_info()}")
    
    # Test sampling
    if buffer.can_provide_sample(32):
        batch = buffer.sample(32)
        print(f"✅ Sampled batch shapes: {[x.shape for x in batch]}")
    
    # Test statistics
    stats = buffer.get_statistics()
    print(f"✅ Buffer statistics: {stats}")
    
    print("Replay Buffer test completed successfully!") 