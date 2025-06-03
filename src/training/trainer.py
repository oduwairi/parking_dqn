"""
Main Training Loop for DQN Autonomous Parking
Orchestrates all training components according to the research paper methodology.

Features:
- Complete training pipeline with paper-specified hyperparameters
- Target network updates (soft and hard)
- Experience replay and batch training
- Performance monitoring and early stopping
- Automatic checkpointing and model saving
- Training resumption from checkpoints
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.environment import ParkingEnv
from src.dqn import DQNAgent

# Try relative imports first, fall back to absolute
try:
    from .config import TrainingConfig, get_config
    from .logger import TrainingLogger
    from .checkpoint import ModelCheckpoint
except ImportError:
    # When run as main script, use absolute imports
    from src.training.config import TrainingConfig, get_config
    from src.training.logger import TrainingLogger
    from src.training.checkpoint import ModelCheckpoint


class DQNTrainer:
    """
    Complete DQN training pipeline for autonomous parking.
    
    Implements the training methodology from the research paper with
    comprehensive monitoring, checkpointing, and performance tracking.
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        experiment_name: str = None,
        resume_from_checkpoint: str = None
    ):
        """
        Initialize DQN trainer.
        
        Args:
            config: Training configuration (uses paper baseline if None)
            experiment_name: Experiment identifier
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Set up configuration
        self.config = config or get_config("paper_baseline")
        self.experiment_name = experiment_name or f"dqn_parking_{int(time.time())}"
        
        # Set random seeds for reproducibility
        self._set_random_seeds(self.config.random_seed)
        
        # Initialize components
        self.device = self._setup_device()
        self.environment = self._create_environment()
        self.agent = self._create_agent()
        
        # Initialize logging and checkpointing
        self.logger = TrainingLogger(
            experiment_name=self.experiment_name,
            save_frequency=self.config.log_frequency,
            plot_frequency=self.config.evaluation_frequency
        )
        
        self.checkpoint_manager = ModelCheckpoint(
            experiment_name=self.experiment_name,
            keep_best=3,
            keep_latest=5
        )
        
        # Training state
        self.current_episode = 0
        self.training_step = 0
        self.best_performance = float('-inf')
        self.episodes_without_improvement = 0
        
        # Performance tracking
        self.recent_success_rate = 0.0
        self.recent_collision_rate = 1.0
        self.recent_avg_reward = float('-inf')
        
        # Save configuration
        self.logger.save_config(self.config.to_dict())
        
        print(f"🚀 DQN Trainer initialized:")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Device: {self.device}")
        print(f"   Configuration: {type(self.config).__name__}")
        print(self.config)
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print(f"✅ Using CPU")
        return device
    
    def _create_environment(self) -> ParkingEnv:
        """Create training environment."""
        env = ParkingEnv(
            width=self.config.environment_width,
            height=self.config.environment_height,
            max_steps=self.config.max_steps_per_episode,
            enable_obstacles=self.config.enable_obstacles,
            randomize_target=self.config.randomize_target,
            randomize_obstacles=self.config.randomize_obstacles
        )
        
        print(f"✅ Environment created: {self.config.environment_width}×{self.config.environment_height}m")
        return env
    
    def _create_agent(self) -> DQNAgent:
        """Create DQN agent."""
        agent = DQNAgent(
            state_dim=12,
            action_dim=7,
            learning_rate=self.config.learning_rate,
            gamma=self.config.discount_factor,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay=0.995,  # Will be overridden by config.get_epsilon()
            batch_size=self.config.batch_size,
            replay_buffer_size=self.config.replay_buffer_size,
            target_update_freq=self.config.target_update_frequency,
            soft_update_tau=self.config.soft_update_rate,
            use_double_dqn=self.config.double_dqn,
            use_prioritized_replay=self.config.prioritized_replay,
            device=self.device,
            seed=self.config.random_seed
        )
        
        agent_info = agent.get_agent_info()
        print(f"✅ DQN Agent created: {agent_info['network_parameters']:,} parameters")
        
        return agent
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint(
            self.agent, checkpoint_path, device=self.device
        )
        
        # Restore training state
        training_state = checkpoint_data['training_state']
        self.current_episode = training_state.get('episode', 0)
        self.training_step = training_state.get('training_step', 0)
        
        # Restore performance tracking
        performance = checkpoint_data['performance']
        self.best_performance = performance.get('score', float('-inf'))
        
        print(f"✅ Resumed from episode {self.current_episode}, step {self.training_step}")
    
    def train(self, episodes: int = None, render_during_training: bool = False, render_frequency: int = 1, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute complete training process.
        
        Args:
            episodes: Number of episodes to train (overrides config if provided)
            render_during_training: Whether to render episodes during training
            render_frequency: How often to render (1 = every episode, 2 = every other episode, etc.)
            verbose: Whether to print detailed progress
        
        Returns:
            Training results and final performance metrics
        """
        total_episodes = episodes if episodes is not None else self.config.total_episodes
        
        if verbose:
            print(f"\n🎯 Starting DQN Training for {total_episodes:,} episodes")
            print(f"   Target: {self.config.target_success_rate:.0%} success rate, <{self.config.target_collision_rate:.1%} collision rate")
            if render_during_training:
                print(f"   🎮 Visualization: ON (every {render_frequency} episode{'s' if render_frequency > 1 else ''})")
        
        training_start_time = time.time()
        
        try:
            # Main training loop
            for episode in range(self.current_episode + 1, total_episodes + 1):
                self.current_episode = episode
                
                # Determine if we should render this episode
                should_render = render_during_training and (episode % render_frequency == 0)
                
                # Run single episode
                episode_metrics = self._train_episode(render=should_render)
                
                # Update performance tracking
                self._update_performance_tracking(episode_metrics)
                
                # Log episode metrics
                if verbose:
                    self._log_episode(episode_metrics)
                
                # Evaluate and checkpoint at specified intervals
                if episode % self.config.evaluation_frequency == 0:
                    eval_metrics = self._evaluate()
                    is_best = self._is_best_performance(eval_metrics)
                    
                    # Save checkpoint
                    self._save_checkpoint(eval_metrics, is_best)
                    
                    # Check early stopping
                    if self._should_early_stop(eval_metrics):
                        if verbose:
                            print(f"🔴 Early stopping triggered at episode {episode}")
                        break
                
                # Check target performance achieved
                if self._target_performance_achieved():
                    if verbose:
                        print(f"🎯 Target performance achieved at episode {episode}!")
                    break
            
            # Training completed
            training_time = time.time() - training_start_time
            final_metrics = self._finalize_training(training_time)
            
            return final_metrics
            
        except KeyboardInterrupt:
            if verbose:
                print(f"\n⚠️ Training interrupted by user at episode {self.current_episode}")
            final_metrics = self._finalize_training(time.time() - training_start_time)
            return final_metrics
        
        except Exception as e:
            if verbose:
                print(f"\n❌ Training failed: {e}")
            raise
    
    def _train_episode(self, render: bool = False) -> Dict[str, Any]:
        """
        Train a single episode.
        
        Args:
            render: Whether to render this episode
        
        Returns:
            Episode metrics dictionary
        """
        # Reset environment
        state = self.environment.reset()
        total_reward = 0.0
        steps = 0
        episode_start_time = time.time()
        
        # Episode loop
        while steps < self.config.max_steps_per_episode:
            # Get current epsilon for exploration
            epsilon = self.config.get_epsilon(self.current_episode)
            
            # Select action
            action = self.agent.select_action(state, epsilon=epsilon)
            
            # Take step in environment
            next_state, reward, done, info = self.environment.step(action)
            
            # Render if requested
            if render:
                self.environment.render()
                time.sleep(0.05)  # Small delay to make visualization visible
            
            # Clip reward if configured
            if self.config.reward_clipping:
                reward = np.clip(reward, -1.0, 1.0)
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent if enough experiences collected
            if self.agent.can_train() and self.agent.replay_buffer.size >= self.config.min_replay_size:
                training_metrics = self.agent.train_step()
                
                # Apply gradient clipping if configured
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.main_network.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # Log training metrics
                if training_metrics:
                    self._log_training_step(training_metrics, epsilon)
                
                self.training_step += 1
            
            # Update target network
            self._update_target_network()
            
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Check episode termination
            if done:
                break
        
        # Calculate episode metrics
        episode_time = time.time() - episode_start_time
        success = info.get('is_successful', False)
        collision = info.get('is_collision', False)
        timeout = steps >= self.config.max_steps_per_episode and not done
        distance_to_target = info.get('distance_to_target', float('inf'))
        
        # Calculate parking accuracy if successful
        parking_accuracy = 0.0
        if success and 'parking_accuracy' in info:
            position_acc = info['parking_accuracy'].get('position_accuracy', 0.0)
            orientation_acc = info['parking_accuracy'].get('orientation_accuracy', 0.0)
            parking_accuracy = (position_acc + orientation_acc) / 2.0
        
        episode_metrics = {
            'episode': self.current_episode,
            'total_reward': total_reward,
            'steps': steps,
            'success': success,
            'collision': collision,
            'timeout': timeout,
            'distance_to_target': distance_to_target,
            'parking_accuracy': parking_accuracy,
            'exploration_rate': epsilon,
            'episode_time': episode_time,
            'training_step': self.training_step
        }
        
        return episode_metrics
    
    def _update_target_network(self):
        """Update target network according to configuration."""
        if self.config.use_soft_update:
            # Soft update every training step using standard PyTorch operations
            if self.training_step > 0:
                for target_param, main_param in zip(
                    self.agent.target_network.parameters(), 
                    self.agent.main_network.parameters()
                ):
                    target_param.data.copy_(
                        self.config.soft_update_rate * main_param.data + 
                        (1.0 - self.config.soft_update_rate) * target_param.data
                    )
        else:
            # Hard update every N steps using state_dict
            if self.training_step % self.config.target_update_frequency == 0:
                self.agent.target_network.load_state_dict(self.agent.main_network.state_dict())
    
    def _log_episode(self, episode_metrics: Dict[str, Any]):
        """Log episode metrics."""
        self.logger.log_episode(
            episode=episode_metrics['episode'],
            total_reward=episode_metrics['total_reward'],
            steps=episode_metrics['steps'],
            success=episode_metrics['success'],
            collision=episode_metrics['collision'],
            timeout=episode_metrics['timeout'],
            distance_to_target=episode_metrics['distance_to_target'],
            parking_accuracy=episode_metrics['parking_accuracy'],
            exploration_rate=episode_metrics['exploration_rate']
        )
    
    def _log_training_step(self, training_metrics: Dict[str, Any], epsilon: float):
        """Log training step metrics."""
        self.logger.log_training_step(
            training_step=self.training_step,
            loss=training_metrics.get('loss', 0.0),
            mean_q_value=training_metrics.get('mean_q_value', 0.0),
            max_q_value=training_metrics.get('max_q_value', 0.0),
            gradient_norm=training_metrics.get('gradient_norm', 0.0),
            epsilon=epsilon,
            learning_rate=self.agent.optimizer.param_groups[0]['lr']
        )
    
    def _evaluate(self) -> Dict[str, Any]:
        """
        Evaluate current agent performance.
        
        Returns:
            Evaluation metrics
        """
        print(f"🔍 Evaluating agent performance...")
        
        # Run evaluation episodes
        eval_rewards = []
        eval_successes = []
        eval_collisions = []
        eval_steps = []
        
        for eval_ep in range(self.config.evaluation_episodes):
            state = self.environment.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < self.config.max_steps_per_episode:
                # Use greedy policy (no exploration) for evaluation
                action = self.agent.select_action(state, epsilon=0.0)
                state, reward, done, info = self.environment.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_successes.append(info.get('is_successful', False))
            eval_collisions.append(info.get('is_collision', False))
            eval_steps.append(steps)
        
        # Calculate evaluation metrics
        eval_metrics = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'success_rate': np.mean(eval_successes),
            'collision_rate': np.mean(eval_collisions),
            'avg_steps': np.mean(eval_steps),
            'episode': self.current_episode
        }
        
        print(f"   Avg Reward: {eval_metrics['avg_reward']:+.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"   Success Rate: {eval_metrics['success_rate']:.1%}")
        print(f"   Collision Rate: {eval_metrics['collision_rate']:.1%}")
        
        return eval_metrics
    
    def _update_performance_tracking(self, episode_metrics: Dict[str, Any]):
        """Update performance tracking for early stopping."""
        # Update episode statistics for recent performance
        self.agent.update_episode_stats(
            episode_metrics['total_reward'],
            episode_metrics['success']
        )
        
        # Get recent performance from agent
        agent_info = self.agent.get_agent_info()
        self.recent_avg_reward = agent_info.get('average_reward', float('-inf'))
    
    def _is_best_performance(self, eval_metrics: Dict[str, Any]) -> bool:
        """Check if current performance is the best so far."""
        # Composite performance score: prioritize success rate, then reward
        performance_score = (
            eval_metrics['success_rate'] * 100 +  # Success rate weight
            eval_metrics['avg_reward'] * 0.1 -     # Reward weight
            eval_metrics['collision_rate'] * 50    # Collision penalty
        )
        
        is_best = performance_score > self.best_performance
        if is_best:
            self.best_performance = performance_score
            self.episodes_without_improvement = 0
            print(f"🌟 New best performance: {performance_score:.2f}")
        else:
            self.episodes_without_improvement += self.config.evaluation_frequency
        
        return is_best
    
    def _save_checkpoint(self, eval_metrics: Dict[str, Any], is_best: bool):
        """Save training checkpoint."""
        performance_score = (
            eval_metrics['success_rate'] * 100 +
            eval_metrics['avg_reward'] * 0.1 -
            eval_metrics['collision_rate'] * 50
        )
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            agent=self.agent,
            episode=self.current_episode,
            training_step=self.training_step,
            performance_score=performance_score,
            metrics=eval_metrics,
            is_best=is_best
        )
        
        return checkpoint_path
    
    def _should_early_stop(self, eval_metrics: Dict[str, Any]) -> bool:
        """Check if training should be stopped early."""
        # Early stopping conditions
        if self.episodes_without_improvement >= self.config.early_stopping_patience:
            print(f"📊 No improvement for {self.episodes_without_improvement} episodes")
            return True
        
        return False
    
    def _target_performance_achieved(self) -> bool:
        """Check if target performance has been achieved."""
        agent_info = self.agent.get_agent_info()
        
        # Check if we have enough recent data
        if agent_info.get('episode_count', 0) < 100:
            return False
        
        # Get recent performance (would need to implement this in agent)
        # For now, use a simple check
        return (
            self.recent_success_rate >= self.config.target_success_rate and
            self.recent_collision_rate <= self.config.target_collision_rate
        )
    
    def _finalize_training(self, training_time: float) -> Dict[str, Any]:
        """Finalize training and save results."""
        print(f"\n🏁 Training completed!")
        print(f"   Total time: {training_time/3600:.1f} hours")
        print(f"   Episodes: {self.current_episode:,}")
        print(f"   Training steps: {self.training_step:,}")
        
        # Final evaluation
        final_eval = self._evaluate()
        
        # Save final model
        final_model_path = self.checkpoint_manager.save_final_model(
            self.agent, final_eval
        )
        
        # Get training summary
        training_summary = self.logger.get_summary()
        
        # Close logger
        self.logger.close()
        
        # Close environment
        self.environment.close()
        
        # Return final metrics
        final_metrics = {
            'training_completed': True,
            'total_episodes': self.current_episode,
            'total_training_steps': self.training_step,
            'training_time_hours': training_time / 3600,
            'final_evaluation': final_eval,
            'training_summary': training_summary,
            'final_model_path': final_model_path,
            'experiment_name': self.experiment_name
        }
        
        return final_metrics


def main():
    """Main training script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN for Autonomous Parking')
    parser.add_argument('--config', default='paper_baseline', 
                       choices=['paper_baseline', 'quick_test', 'high_performance', 'conservative'],
                       help='Training configuration preset')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override number of training episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    if args.episodes:
        config.total_episodes = args.episodes
    
    print(f"🎮 Starting DQN Training with {args.config} configuration")
    
    # Create and run trainer
    trainer = DQNTrainer(
        config=config,
        experiment_name=args.experiment_name,
        resume_from_checkpoint=args.resume
    )
    
    # Start training
    results = trainer.train()
    
    # Print final results
    print(f"\n📊 Training Results:")
    print(f"   Success Rate: {results['final_evaluation']['success_rate']:.1%}")
    print(f"   Collision Rate: {results['final_evaluation']['collision_rate']:.1%}")
    print(f"   Average Reward: {results['final_evaluation']['avg_reward']:+.2f}")
    print(f"   Final Model: {results['final_model_path']}")


if __name__ == "__main__":
    main() 