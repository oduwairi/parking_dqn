"""
Training Logger and Metrics Tracking
Provides comprehensive logging and monitoring for DQN training progress.

Features:
- Episode metrics tracking (rewards, success rates, collision rates)
- Training performance monitoring (loss, Q-values, epsilon)
- Real-time progress visualization
- CSV and JSON data export
- Training curve plotting
"""

import os
import time
import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features disabled.")


@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    episode: int
    total_reward: float
    steps: int
    success: bool
    collision: bool
    timeout: bool
    distance_to_target: float
    parking_accuracy: float
    exploration_rate: float
    timestamp: float


@dataclass
class TrainingMetrics:
    """Metrics for a training step."""
    training_step: int
    loss: float
    mean_q_value: float
    max_q_value: float
    gradient_norm: float
    epsilon: float
    learning_rate: float
    timestamp: float


class TrainingLogger:
    """
    Comprehensive training logger for DQN autonomous parking.
    
    Tracks and visualizes training progress, performance metrics,
    and provides data export capabilities.
    """
    
    def __init__(
        self, 
        log_dir: str = "data/training_logs",
        experiment_name: str = None,
        save_frequency: int = 100,
        plot_frequency: int = 50
    ):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
            save_frequency: Save data every N episodes
            plot_frequency: Update plots every N episodes
        """
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        self.plot_frequency = plot_frequency
        
        # Create experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dqn_parking_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize data storage
        self.episode_metrics: List[EpisodeMetrics] = []
        self.training_metrics: List[TrainingMetrics] = []
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)  # Recent rewards for moving average
        self.success_buffer = deque(maxlen=100)   # Recent success for success rate
        self.collision_buffer = deque(maxlen=100) # Recent collisions for collision rate
        
        # Training timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Best performance tracking
        self.best_avg_reward = float('-inf')
        self.best_success_rate = 0.0
        self.best_episode = 0
        
        # File paths
        self.episode_csv = os.path.join(self.experiment_dir, "episodes.csv")
        self.training_csv = os.path.join(self.experiment_dir, "training.csv")
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        self.summary_file = os.path.join(self.experiment_dir, "summary.json")
        
        # Initialize CSV files
        self._init_csv_files()
        
        print(f"🗂️ Training logger initialized: {self.experiment_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Episode metrics CSV
        with open(self.episode_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'total_reward', 'steps', 'success', 'collision', 'timeout',
                'distance_to_target', 'parking_accuracy', 'exploration_rate', 'timestamp'
            ])
        
        # Training metrics CSV
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'training_step', 'loss', 'mean_q_value', 'max_q_value',
                'gradient_norm', 'epsilon', 'learning_rate', 'timestamp'
            ])
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        success: bool,
        collision: bool,
        timeout: bool,
        distance_to_target: float,
        parking_accuracy: float = 0.0,
        exploration_rate: float = 0.0
    ):
        """
        Log metrics for a completed episode.
        
        Args:
            episode: Episode number
            total_reward: Cumulative reward for episode
            steps: Number of steps taken
            success: Whether parking was successful
            collision: Whether collision occurred
            timeout: Whether episode timed out
            distance_to_target: Final distance to parking target
            parking_accuracy: Accuracy of parking (0-1)
            exploration_rate: Current epsilon value
        """
        metrics = EpisodeMetrics(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            success=success,
            collision=collision,
            timeout=timeout,
            distance_to_target=distance_to_target,
            parking_accuracy=parking_accuracy,
            exploration_rate=exploration_rate,
            timestamp=time.time()
        )
        
        self.episode_metrics.append(metrics)
        
        # Update performance tracking
        self.episode_rewards.append(total_reward)
        self.success_buffer.append(success)
        self.collision_buffer.append(collision)
        
        # Save to CSV
        with open(self.episode_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, total_reward, steps, success, collision, timeout,
                distance_to_target, parking_accuracy, exploration_rate, metrics.timestamp
            ])
        
        # Log progress at specified frequency
        if episode % self.save_frequency == 0 or episode == 1:
            self._log_progress(episode)
        
        # Update plots at specified frequency
        if MATPLOTLIB_AVAILABLE and episode % self.plot_frequency == 0:
            self._update_plots()
    
    def log_training_step(
        self,
        training_step: int,
        loss: float,
        mean_q_value: float,
        max_q_value: float,
        gradient_norm: float,
        epsilon: float,
        learning_rate: float
    ):
        """
        Log metrics for a training step.
        
        Args:
            training_step: Training step number
            loss: Training loss value
            mean_q_value: Mean Q-value across batch
            max_q_value: Maximum Q-value in batch
            gradient_norm: Gradient norm magnitude
            epsilon: Current exploration rate
            learning_rate: Current learning rate
        """
        metrics = TrainingMetrics(
            training_step=training_step,
            loss=loss,
            mean_q_value=mean_q_value,
            max_q_value=max_q_value,
            gradient_norm=gradient_norm,
            epsilon=epsilon,
            learning_rate=learning_rate,
            timestamp=time.time()
        )
        
        self.training_metrics.append(metrics)
        
        # Save to CSV
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                training_step, loss, mean_q_value, max_q_value,
                gradient_norm, epsilon, learning_rate, metrics.timestamp
            ])
    
    def _log_progress(self, episode: int):
        """Log current training progress to console."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        time_since_last = current_time - self.last_log_time
        
        # Calculate performance metrics
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        success_rate = np.mean(self.success_buffer) if self.success_buffer else 0.0
        collision_rate = np.mean(self.collision_buffer) if self.collision_buffer else 0.0
        
        # Update best performance
        if avg_reward > self.best_avg_reward:
            self.best_avg_reward = avg_reward
            self.best_episode = episode
        
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
        
        # Calculate episodes per second
        eps_per_sec = self.save_frequency / time_since_last if time_since_last > 0 else 0
        
        print(f"\n📊 Episode {episode:,} Progress Report:")
        print(f"   Time: {elapsed_time/60:.1f}m | {eps_per_sec:.1f} eps/sec")
        print(f"   Avg Reward (100): {avg_reward:+.2f}")
        print(f"   Success Rate: {success_rate:.1%} | Collision Rate: {collision_rate:.1%}")
        print(f"   Best: {self.best_avg_reward:+.2f} @ episode {self.best_episode}")
        
        if self.training_metrics:
            latest_training = self.training_metrics[-1]
            print(f"   Training: Loss={latest_training.loss:.4f}, Q̄={latest_training.mean_q_value:.3f}, ε={latest_training.epsilon:.3f}")
        
        self.last_log_time = current_time
    
    def _update_plots(self):
        """Update training progress plots."""
        if not MATPLOTLIB_AVAILABLE or len(self.episode_metrics) < 2:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Training Progress: {self.experiment_name}")
            
            episodes = [m.episode for m in self.episode_metrics]
            rewards = [m.total_reward for m in self.episode_metrics]
            successes = [m.success for m in self.episode_metrics]
            collisions = [m.collision for m in self.episode_metrics]
            
            # Plot 1: Episode rewards
            axes[0, 0].plot(episodes, rewards, alpha=0.6, color='blue')
            if len(rewards) >= 10:
                # Moving average
                window = min(50, len(rewards) // 4)
                rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes_ma = episodes[window-1:]
                axes[0, 0].plot(episodes_ma, rewards_ma, color='red', linewidth=2, label=f'MA({window})')
                axes[0, 0].legend()
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Success/Collision rates
            window = min(100, len(successes) // 2)
            if len(successes) >= window:
                success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
                collision_rate = np.convolve(collisions, np.ones(window)/window, mode='valid')
                episodes_rate = episodes[window-1:]
                
                axes[0, 1].plot(episodes_rate, success_rate, color='green', label='Success Rate')
                axes[0, 1].plot(episodes_rate, collision_rate, color='red', label='Collision Rate')
                axes[0, 1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target Success')
                axes[0, 1].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Target Collision')
                axes[0, 1].set_title(f'Performance Rates (MA{window})')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Rate')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_ylim(0, 1)
            
            # Plot 3: Training loss
            if self.training_metrics:
                training_steps = [m.training_step for m in self.training_metrics]
                losses = [m.loss for m in self.training_metrics]
                axes[1, 0].plot(training_steps, losses, color='orange', alpha=0.7)
                axes[1, 0].set_title('Training Loss')
                axes[1, 0].set_xlabel('Training Step')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Q-values and Epsilon
            if self.training_metrics:
                q_values = [m.mean_q_value for m in self.training_metrics]
                epsilons = [m.epsilon for m in self.training_metrics]
                
                ax4 = axes[1, 1]
                ax4_twin = ax4.twinx()
                
                line1 = ax4.plot(training_steps, q_values, color='purple', label='Mean Q-value')
                line2 = ax4_twin.plot(training_steps, epsilons, color='brown', label='Epsilon')
                
                ax4.set_xlabel('Training Step')
                ax4.set_ylabel('Mean Q-value', color='purple')
                ax4_twin.set_ylabel('Epsilon', color='brown')
                ax4.set_title('Q-values and Exploration')
                ax4.grid(True, alpha=0.3)
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper right')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.experiment_dir, "training_progress.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to update plots: {e}")
    
    def save_config(self, config: Dict[str, Any]):
        """Save training configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.episode_metrics:
            return {}
        
        rewards = [m.total_reward for m in self.episode_metrics]
        successes = [m.success for m in self.episode_metrics]
        collisions = [m.collision for m in self.episode_metrics]
        steps = [m.steps for m in self.episode_metrics]
        
        # Recent performance (last 100 episodes)
        recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        recent_successes = successes[-100:] if len(successes) >= 100 else successes
        recent_collisions = collisions[-100:] if len(collisions) >= 100 else collisions
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_metrics),
            'total_training_steps': len(self.training_metrics),
            'training_time_hours': (time.time() - self.start_time) / 3600,
            
            # Overall performance
            'overall_stats': {
                'avg_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'success_rate': float(np.mean(successes)),
                'collision_rate': float(np.mean(collisions)),
                'avg_steps': float(np.mean(steps))
            },
            
            # Recent performance
            'recent_stats': {
                'avg_reward': float(np.mean(recent_rewards)),
                'success_rate': float(np.mean(recent_successes)),
                'collision_rate': float(np.mean(recent_collisions))
            },
            
            # Best performance
            'best_performance': {
                'best_avg_reward': self.best_avg_reward,
                'best_episode': self.best_episode,
                'best_success_rate': self.best_success_rate
            }
        }
        
        return summary
    
    def save_summary(self):
        """Save training summary to file."""
        summary = self.get_summary()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary
    
    def close(self):
        """Close logger and save final summary."""
        print(f"\n🏁 Training completed! Saving final summary...")
        summary = self.save_summary()
        
        if MATPLOTLIB_AVAILABLE:
            self._update_plots()
        
        print(f"📁 Experiment data saved to: {self.experiment_dir}")
        print(f"📊 Final Summary:")
        print(f"   Episodes: {summary.get('total_episodes', 0):,}")
        print(f"   Training Time: {summary.get('training_time_hours', 0):.1f} hours")
        
        recent = summary.get('recent_stats', {})
        print(f"   Recent Performance:")
        print(f"     Avg Reward: {recent.get('avg_reward', 0):+.2f}")
        print(f"     Success Rate: {recent.get('success_rate', 0):.1%}")
        print(f"     Collision Rate: {recent.get('collision_rate', 0):.1%}")


if __name__ == "__main__":
    # Example usage and testing
    logger = TrainingLogger(experiment_name="test_experiment")
    
    # Simulate some training data
    for episode in range(1, 51):
        # Simulate episode metrics
        reward = np.random.normal(10, 20)  # Random reward
        success = np.random.random() < 0.3  # 30% success rate initially
        collision = np.random.random() < 0.1  # 10% collision rate
        
        logger.log_episode(
            episode=episode,
            total_reward=reward,
            steps=np.random.randint(50, 200),
            success=success,
            collision=collision,
            timeout=not (success or collision),
            distance_to_target=np.random.uniform(1, 10),
            parking_accuracy=np.random.uniform(0, 1),
            exploration_rate=max(0.01, 1.0 - episode * 0.02)
        )
        
        # Simulate training metrics
        if episode % 2 == 0:  # Training happens every other episode
            logger.log_training_step(
                training_step=episode // 2,
                loss=np.random.uniform(0.1, 1.0),
                mean_q_value=np.random.uniform(-10, 10),
                max_q_value=np.random.uniform(0, 20),
                gradient_norm=np.random.uniform(0.1, 2.0),
                epsilon=max(0.01, 1.0 - episode * 0.02),
                learning_rate=1e-3
            )
    
    logger.close()
    print("✅ Logger test completed successfully!") 