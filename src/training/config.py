"""
Training Configuration and Hyperparameters
Based on research paper specifications for autonomous parking DQN training.

Hyperparameters from paper:
- Learning rate α = 10⁻³
- Discount factor γ = 0.9-0.95
- Batch size B = 64
- Target update frequency N = 1000
- Soft update rate τ = 10⁻³
- Training episodes: 5000
- Epsilon decay: exponential
"""

import math
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Training hyperparameter configuration."""
    
    # Core training parameters
    total_episodes: int = 5000
    max_steps_per_episode: int = 1000
    learning_rate: float = 1e-3
    discount_factor: float = 0.95  # γ (gamma)
    batch_size: int = 64  # B
    
    # Network update parameters
    target_update_frequency: int = 1000  # N (hard update frequency)
    soft_update_rate: float = 1e-3  # τ (tau) for soft updates
    use_soft_update: bool = True  # Use soft updates instead of hard updates
    
    # Epsilon-greedy exploration
    epsilon_start: float = 1.0  # ε_max
    epsilon_end: float = 0.01   # ε_min (never zero to encourage exploration)
    epsilon_decay_episodes: int = 2000  # Episodes to decay from start to end
    epsilon_decay_type: str = "exponential"  # "linear" or "exponential"
    
    # Experience replay
    replay_buffer_size: int = 100000  # ~10^5 transitions
    min_replay_size: int = 1000  # Minimum experiences before training starts
    prioritized_replay: bool = False  # Use prioritized experience replay
    
    # Environment parameters
    environment_width: float = 50.0
    environment_height: float = 30.0
    enable_obstacles: bool = True
    randomize_target: bool = True
    randomize_obstacles: bool = False
    
    # Training stability
    gradient_clip_norm: float = 10.0  # Gradient clipping
    reward_clipping: bool = False  # Clip rewards to [-1, 1]
    double_dqn: bool = True  # Use Double DQN
    
    # Logging and checkpointing
    log_frequency: int = 10  # Log every N episodes
    checkpoint_frequency: int = 100  # Save model every N episodes
    evaluation_frequency: int = 50  # Evaluate performance every N episodes
    evaluation_episodes: int = 10  # Number of episodes for evaluation
    
    # Training monitoring
    early_stopping_patience: int = 500  # Stop if no improvement for N episodes
    target_success_rate: float = 0.7  # Target 70% success rate
    target_collision_rate: float = 0.01  # Target ≤1% collision rate
    
    # Hardware acceleration
    use_gpu: bool = True  # Use CUDA if available
    num_workers: int = 1  # Number of parallel environments
    
    # Reproducibility
    random_seed: int = 42
    
    def get_epsilon(self, episode: int) -> float:
        """
        Calculate epsilon value for given episode using decay strategy.
        
        Args:
            episode: Current episode number
            
        Returns:
            Epsilon value for exploration
        """
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end
        
        progress = episode / self.epsilon_decay_episodes
        
        if self.epsilon_decay_type == "linear":
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        elif self.epsilon_decay_type == "exponential":
            # ε_t = ε_min + (ε_max - ε_min) * exp(-λt)
            decay_rate = -math.log(self.epsilon_end / self.epsilon_start) / self.epsilon_decay_episodes
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-decay_rate * episode)
        else:
            raise ValueError(f"Unknown epsilon decay type: {self.epsilon_decay_type}")
        
        return max(epsilon, self.epsilon_end)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.learning_rate <= 0 or self.learning_rate > 1:
            raise ValueError(f"Learning rate must be in (0, 1], got {self.learning_rate}")
        
        if self.discount_factor <= 0 or self.discount_factor > 1:
            raise ValueError(f"Discount factor must be in (0, 1], got {self.discount_factor}")
        
        if self.epsilon_start < self.epsilon_end:
            raise ValueError(f"Epsilon start ({self.epsilon_start}) must be >= epsilon end ({self.epsilon_end})")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        if self.target_update_frequency <= 0:
            raise ValueError(f"Target update frequency must be positive, got {self.target_update_frequency}")
        
        if self.replay_buffer_size < self.min_replay_size:
            raise ValueError(f"Replay buffer size ({self.replay_buffer_size}) must be >= min replay size ({self.min_replay_size})")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Save configuration to YAML file."""
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["Training Configuration:"]
        lines.append(f"  Episodes: {self.total_episodes}")
        lines.append(f"  Learning Rate: {self.learning_rate}")
        lines.append(f"  Discount Factor: {self.discount_factor}")
        lines.append(f"  Batch Size: {self.batch_size}")
        lines.append(f"  Epsilon: {self.epsilon_start} → {self.epsilon_end} (over {self.epsilon_decay_episodes} episodes)")
        lines.append(f"  Target Update: {'Soft' if self.use_soft_update else 'Hard'} (τ={self.soft_update_rate})")
        lines.append(f"  Replay Buffer: {self.replay_buffer_size:,} transitions")
        lines.append(f"  Double DQN: {self.double_dqn}")
        lines.append(f"  GPU: {self.use_gpu}")
        return "\n".join(lines)


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined training configurations for different scenarios."""
    
    @staticmethod
    def paper_baseline() -> TrainingConfig:
        """Configuration exactly as specified in the research paper."""
        return TrainingConfig(
            total_episodes=5000,
            learning_rate=1e-3,
            discount_factor=0.95,
            batch_size=64,
            target_update_frequency=1000,
            soft_update_rate=1e-3,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_episodes=2000,
            replay_buffer_size=100000,
            double_dqn=True,
            use_soft_update=True
        )
    
    @staticmethod
    def quick_test() -> TrainingConfig:
        """Fast configuration for testing and debugging."""
        return TrainingConfig(
            total_episodes=100,
            max_steps_per_episode=200,
            learning_rate=1e-3,
            discount_factor=0.9,
            batch_size=32,
            target_update_frequency=50,
            epsilon_decay_episodes=50,
            replay_buffer_size=5000,
            min_replay_size=100,
            log_frequency=5,
            checkpoint_frequency=25,
            evaluation_frequency=20,
            evaluation_episodes=3
        )
    
    @staticmethod
    def high_performance() -> TrainingConfig:
        """Configuration optimized for performance and stability."""
        return TrainingConfig(
            total_episodes=7500,
            learning_rate=5e-4,
            discount_factor=0.99,
            batch_size=128,
            target_update_frequency=2000,
            soft_update_rate=5e-4,
            epsilon_decay_episodes=3000,
            replay_buffer_size=200000,
            prioritized_replay=True,
            gradient_clip_norm=1.0,
            early_stopping_patience=1000
        )
    
    @staticmethod
    def conservative() -> TrainingConfig:
        """Conservative configuration for stable training."""
        return TrainingConfig(
            total_episodes=5000,
            learning_rate=1e-4,
            discount_factor=0.95,
            batch_size=32,
            target_update_frequency=500,
            soft_update_rate=1e-4,
            epsilon_start=0.5,
            epsilon_decay_episodes=3000,
            gradient_clip_norm=5.0,
            use_soft_update=True
        )
    
    @staticmethod
    def debug_viz() -> TrainingConfig:
        """Configuration for visual debugging with rendering."""
        return TrainingConfig(
            total_episodes=50,
            max_steps_per_episode=100,
            learning_rate=1e-3,
            discount_factor=0.9,
            batch_size=16,
            target_update_frequency=25,
            epsilon_start=0.8,
            epsilon_end=0.1,
            epsilon_decay_episodes=25,
            replay_buffer_size=2000,
            min_replay_size=50,
            log_frequency=5,
            checkpoint_frequency=10,
            evaluation_frequency=10,
            evaluation_episodes=1,
            enable_obstacles=True,
            randomize_target=False,  # Keep target fixed for easier debugging
            use_gpu=False  # CPU for debugging to avoid GPU memory issues
        )

    @staticmethod
    def progressive_simple() -> TrainingConfig:
        """Stage 1: SPARSE rewards - agent must truly learn parking behavior."""
        return TrainingConfig(
            total_episodes=800,  # Fewer episodes, more focused
            max_steps_per_episode=100,  # Shorter episodes force efficiency
            learning_rate=2e-3,  # Higher LR for sparse rewards
            discount_factor=0.95,  # Higher discount for long-term planning
            batch_size=64,  # Larger batch for stable learning
            target_update_frequency=200,  # More frequent updates
            soft_update_rate=1e-2,  # Faster target network updates
            epsilon_start=1.0,
            epsilon_end=0.05,  # Higher final epsilon for continued exploration
            epsilon_decay_episodes=400,  # Faster decay
            replay_buffer_size=20000,  # Larger buffer for sparse rewards
            min_replay_size=500,  # More experience before training
            log_frequency=20,
            checkpoint_frequency=50,
            evaluation_frequency=25,
            evaluation_episodes=5,
            enable_obstacles=False,  # No obstacles initially
            randomize_target=False,  # Fixed target for learning
            use_gpu=True,
            gradient_clip_norm=1.0,
            early_stopping_patience=200,  # Faster stopping
            environment_width=35.0,  # Even smaller environment
            environment_height=20.0,
            target_success_rate=0.3,  # Lower initial target (30%)
            target_collision_rate=0.1  # Allow some collisions initially
        )
    
    @staticmethod
    def progressive_obstacles() -> TrainingConfig:
        """Stage 2: Add obstacles after basic parking is learned."""
        return TrainingConfig(
            total_episodes=1500,
            max_steps_per_episode=300,
            learning_rate=3e-4,  # Even lower for complex environment
            discount_factor=0.95,
            batch_size=64,
            target_update_frequency=500,
            soft_update_rate=1e-3,
            epsilon_start=0.3,  # Start with less exploration
            epsilon_end=0.02,
            epsilon_decay_episodes=1000,
            replay_buffer_size=50000,
            min_replay_size=500,
            log_frequency=25,
            checkpoint_frequency=100,
            evaluation_frequency=50,
            evaluation_episodes=3,
            enable_obstacles=True,  # Add obstacles
            randomize_target=False,  # Still fixed target
            use_gpu=True,
            gradient_clip_norm=1.0,
            early_stopping_patience=500
        )
    
    @staticmethod
    def progressive_full() -> TrainingConfig:
        """Stage 3: Full complexity with randomized targets."""
        return TrainingConfig(
            total_episodes=2000,
            max_steps_per_episode=400,
            learning_rate=1e-4,  # Very stable learning
            discount_factor=0.95,
            batch_size=128,
            target_update_frequency=1000,
            soft_update_rate=5e-4,
            epsilon_start=0.2,
            epsilon_end=0.01,
            epsilon_decay_episodes=1500,
            replay_buffer_size=100000,
            min_replay_size=1000,
            log_frequency=25,
            checkpoint_frequency=100,
            evaluation_frequency=50,
            evaluation_episodes=5,
            enable_obstacles=True,
            randomize_target=True,  # Full randomization
            use_gpu=True,
            gradient_clip_norm=0.5,
            early_stopping_patience=800,
            target_success_rate=0.7,
            target_collision_rate=0.05
        )


def get_config(preset: str = "paper_baseline") -> TrainingConfig:
    """
    Get a predefined training configuration.
    
    Args:
        preset: Configuration preset name
        
    Returns:
        TrainingConfig instance
        
    Raises:
        ValueError: If preset name is not recognized
    """
    presets = {
        "paper_baseline": ConfigPresets.paper_baseline,
        "quick_test": ConfigPresets.quick_test,
        "high_performance": ConfigPresets.high_performance,
        "conservative": ConfigPresets.conservative,
        "debug_viz": ConfigPresets.debug_viz,
        "progressive_simple": ConfigPresets.progressive_simple,
        "progressive_obstacles": ConfigPresets.progressive_obstacles,
        "progressive_full": ConfigPresets.progressive_full
    }
    
    if preset not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    config = presets[preset]()
    config.validate()
    return config


if __name__ == "__main__":
    # Example usage and testing
    print("Available Configuration Presets:")
    for preset_name in ["paper_baseline", "quick_test", "high_performance", "conservative", "debug_viz", "progressive_simple", "progressive_obstacles", "progressive_full"]:
        print(f"\n{preset_name.upper()}:")
        config = get_config(preset_name)
        print(config)
        
        # Test epsilon decay
        print(f"  Epsilon progression: {config.get_epsilon(0):.3f} → {config.get_epsilon(config.epsilon_decay_episodes//2):.3f} → {config.get_epsilon(config.epsilon_decay_episodes):.3f}") 