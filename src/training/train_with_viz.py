"""
Training script with visualization for debugging agent behavior.
This allows you to watch the agent learn in real-time.

Phase 7 Enhancement: Progressive Training with GPU acceleration
"""

import argparse
import time
import os
import sys
import pygame
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.trainer import DQNTrainer
from src.training.config import get_config


def progressive_training_with_viz(start_stage: int = 0, use_gpu: bool = True, viz_freq: int = 50):
    """
    Run progressive training with periodic visualization.
    
    Args:
        start_stage: Which stage to start from (0=simple, 1=obstacles, 2=full)
        use_gpu: Whether to use GPU acceleration
        viz_freq: Visualization frequency (every N episodes)
    """
    
    # Define progressive stages
    stages = [
        ("progressive_simple", "🟢 Stage 1: Simple Environment (No Obstacles)"),
        ("progressive_obstacles", "🟡 Stage 2: With Obstacles"),
        ("progressive_full", "🔴 Stage 3: Full Complexity")
    ]
    
    print(f"🚀 PROGRESSIVE TRAINING WITH PERIODIC VISUALIZATION")
    print(f"Starting from stage: {start_stage + 1}")
    print(f"GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Visualization: Every {viz_freq} episodes")
    
    # Success criteria for progression
    stage_criteria = {
        0: {"min_success_rate": 0.2, "max_collision_rate": 0.3},  # Simple
        1: {"min_success_rate": 0.4, "max_collision_rate": 0.2},  # Obstacles
        2: {"min_success_rate": 0.6, "max_collision_rate": 0.1}   # Full
    }
    
    previous_model = None
    
    for stage_idx in range(start_stage, len(stages)):
        config_name, description = stages[stage_idx]
        
        print(f"\n{'='*60}")
        print(f"🎯 {description}")
        print(f"{'='*60}")
        
        # Load configuration
        config = get_config(config_name)
        
        # Override GPU setting
        config.use_gpu = use_gpu
        
        # Create stage-specific experiment name
        experiment_name = f"progressive_stage{stage_idx + 1}_{int(time.time())}"
        
        # Create trainer
        trainer = DQNTrainer(
            config=config,
            experiment_name=experiment_name,
            resume_from_checkpoint=previous_model
        )
        
        print(f"📊 Configuration:")
        print(f"   Episodes: {config.total_episodes:,}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Obstacles: {'Yes' if config.enable_obstacles else 'No'}")
        print(f"   Random Targets: {'Yes' if config.randomize_target else 'No'}")
        print(f"   Device: {'GPU' if config.use_gpu else 'CPU'}")
        
        # Get success criteria for this stage
        criteria = stage_criteria.get(stage_idx, {})
        min_success = criteria.get("min_success_rate", 0)
        max_collision = criteria.get("max_collision_rate", 1)
        
        print(f"   Success Criteria: ≥{min_success:.0%} success, ≤{max_collision:.0%} collision")
        print(f"   🎮 Visualization: Every {viz_freq} episodes")
        
        try:
            # Train with periodic visualization
            print(f"\n🚀 Starting training with periodic visualization...")
            print(f"   Watch the pygame window every {viz_freq} episodes!")
            
            results = trainer.train(
                episodes=config.total_episodes,
                render_during_training=True,
                render_frequency=viz_freq,
                verbose=True
            )
            
            # Extract performance
            final_eval = results['final_evaluation']
            success_rate = final_eval['success_rate']
            collision_rate = final_eval['collision_rate']
            avg_reward = final_eval['avg_reward']
            
            print(f"\n📈 STAGE {stage_idx + 1} RESULTS:")
            print(f"   Success Rate: {success_rate:.1%}")
            print(f"   Collision Rate: {collision_rate:.1%}")
            print(f"   Average Reward: {avg_reward:+.2f}")
            
            # Check if stage criteria met
            stage_passed = (success_rate >= min_success and collision_rate <= max_collision)
            
            if stage_passed:
                print(f"✅ Stage {stage_idx + 1} PASSED! Criteria met.")
                previous_model = results['final_model_path']
                
                if stage_idx < len(stages) - 1:
                    print(f"🔄 Proceeding to next stage...")
                    time.sleep(2)  # Brief pause
                else:
                    print(f"🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
                    
            else:
                print(f"❌ Stage {stage_idx + 1} criteria not met:")
                print(f"   Required: Success ≥{min_success:.1%}, Collision ≤{max_collision:.1%}")
                print(f"   Achieved: Success {success_rate:.1%}, Collision {collision_rate:.1%}")
                
                # Ask user if they want to continue
                response = input("\nContinue to next stage anyway? (y/n): ").lower()
                if response == 'y':
                    previous_model = results['final_model_path']
                    print(f"🔄 Continuing to next stage...")
                else:
                    print(f"🛑 Training stopped by user")
                    break
                    
        except KeyboardInterrupt:
            print(f"\n⚠️ Stage {stage_idx + 1} interrupted by user")
            response = input("Continue to next stage? (y/n): ").lower()
            if response != 'y':
                print(f"🛑 Training stopped")
                break
        
        except Exception as e:
            print(f"\n❌ Stage {stage_idx + 1} failed: {e}")
            response = input("Continue to next stage anyway? (y/n): ").lower()
            if response != 'y':
                print(f"🛑 Training stopped due to error")
                break
    
    print(f"\n🏁 Progressive training session ended")


def single_stage_training_with_viz(stage: str, use_gpu: bool = True, 
                                  render_frequency: int = 50, enable_rendering: bool = True):
    """Train a single stage with configurable visualization."""
    
    print(f"🎯 Single Stage Training: {stage}")
    print(f"GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Visualization: {'Every {render_frequency} episodes' if enable_rendering else 'Disabled'}")
    
    # Load configuration
    config = get_config(stage)
    config.use_gpu = use_gpu
    
    # Create trainer
    experiment_name = f"single_{stage}_{int(time.time())}"
    trainer = DQNTrainer(config=config, experiment_name=experiment_name)
    
    print(f"\n📊 Configuration:")
    print(f"   Episodes: {config.total_episodes:,}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Device: {'GPU' if config.use_gpu else 'CPU'}")
    print(f"   🎮 Visualization: {'ON' if enable_rendering else 'OFF'} (every {render_frequency} episodes)")
    
    try:
        # Start training with configurable visualization
        if enable_rendering:
            print(f"\n🚀 Starting training with periodic visualization...")
            print(f"   Watch the pygame window every {render_frequency} episodes!")
        else:
            print(f"\n🚀 Starting fast training (no visualization)...")
            print(f"   Training optimized for maximum speed!")
        
        results = trainer.train(
            episodes=config.total_episodes,
            render_during_training=enable_rendering,
            render_frequency=render_frequency,
            verbose=True
        )
        
        print(f"\n📊 Final Results:")
        final_eval = results['final_evaluation']
        print(f"   Success Rate: {final_eval['success_rate']:.1%}")
        print(f"   Collision Rate: {final_eval['collision_rate']:.1%}")
        print(f"   Average Reward: {final_eval['avg_reward']:+.2f}")
        print(f"   Model saved: {results['final_model_path']}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise e


def resume_training_with_viz(checkpoint_path: str, additional_episodes: int = 500,
                           render_frequency: int = 50, enable_rendering: bool = True,
                           config_name: str = None):
    """Resume training from a specific checkpoint."""
    
    print(f"🔄 Resuming Training from Checkpoint")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Additional Episodes: {additional_episodes}")
    print(f"Visualization: {'Every ' + str(render_frequency) + ' episodes' if enable_rendering else 'Disabled'}")
    
    # Validate checkpoint path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Try to detect config from checkpoint or use provided config_name
    if config_name is None:
        # Try to infer config from checkpoint directory or filename
        if "progressive_simple" in checkpoint_path:
            config_name = "progressive_simple"
        elif "progressive_obstacles" in checkpoint_path:
            config_name = "progressive_obstacles" 
        elif "progressive_full" in checkpoint_path:
            config_name = "progressive_full"
        else:
            # Default to progressive_simple if can't detect
            config_name = "progressive_simple"
            print(f"⚠️ Could not detect config from checkpoint path, using default: {config_name}")
    
    print(f"Using config: {config_name}")
    
    # Load configuration
    config = get_config(config_name)
    config.use_gpu = True
    
    # Override total episodes to be additional episodes from current checkpoint
    # The trainer will start from the checkpoint episode and run for additional_episodes more
    config.total_episodes = additional_episodes
    
    # Create trainer with resume
    experiment_name = f"resumed_{int(time.time())}"
    trainer = DQNTrainer(
        config=config,
        experiment_name=experiment_name,
        resume_from_checkpoint=checkpoint_path
    )
    
    print(f"\n📊 Resume Configuration:")
    print(f"   Config: {config_name}")
    print(f"   Additional Episodes: {additional_episodes}")
    print(f"   Device: {'GPU' if config.use_gpu else 'CPU'}")
    print(f"   🎮 Visualization: {'ON' if enable_rendering else 'OFF'} (every {render_frequency} episodes)")
    
    try:
        results = trainer.train(
            episodes=additional_episodes,
            render_during_training=enable_rendering,
            render_frequency=render_frequency,
            verbose=True
        )
        
        print(f"\n📊 Resume Results:")
        final_eval = results['final_evaluation']
        print(f"   Success Rate: {final_eval['success_rate']:.1%}")
        print(f"   Collision Rate: {final_eval['collision_rate']:.1%}")
        print(f"   Average Reward: {final_eval['avg_reward']:+.2f}")
        print(f"   Final Model: {results['final_model_path']}")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Resume failed: {e}")
        raise e


def main():
    """Main training execution with argument parsing."""
    parser = argparse.ArgumentParser(description='Progressive DQN Training with Visualization')
    parser.add_argument('--mode', choices=['progressive', 'single', 'resume'], default='single',
                       help='Training mode: progressive (3 stages), single stage, or resume')
    parser.add_argument('--stage', choices=['progressive_simple', 'progressive_obstacles', 'progressive_full', 'debug_viz'], 
                       default='progressive_simple', help='Single stage to train')
    parser.add_argument('--start-stage', type=int, default=0, choices=[0, 1, 2],
                       help='Which stage to start progressive training from (0=simple, 1=obstacles, 2=full)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage (disable GPU)')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization for maximum speed')
    parser.add_argument('--viz-freq', type=int, default=50, help='Visualization frequency (every N episodes)')
    parser.add_argument('--resume-path', type=str, help='Path to checkpoint for resuming training')
    parser.add_argument('--additional-episodes', type=int, default=500, help='Additional episodes when resuming')
    parser.add_argument('--config-name', type=str, choices=['progressive_simple', 'progressive_obstacles', 'progressive_full'], 
                       help='Config name to use when resuming (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    use_gpu = not args.cpu
    enable_rendering = not args.no_viz
    
    print(f"🎮 Phase 7: Progressive DQN Training with Visualization")
    print(f"Mode: {args.mode}")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Speed Mode: {'🚀 FAST' if not enable_rendering else f'🎮 VISUAL (every {args.viz_freq} episodes)'}")
    
    if args.mode == 'progressive':
        # Progressive training
        progressive_training_with_viz(
            start_stage=args.start_stage,
            use_gpu=use_gpu,
            viz_freq=args.viz_freq
        )
    elif args.mode == 'resume':
        # Resume training
        if not args.resume_path:
            print("❌ --resume-path required for resume mode")
            return
        resume_training_with_viz(
            checkpoint_path=args.resume_path,
            additional_episodes=args.additional_episodes,
            render_frequency=args.viz_freq,
            enable_rendering=enable_rendering,
            config_name=args.config_name
        )
    else:
        # Single stage training
        single_stage_training_with_viz(
            stage=args.stage, 
            use_gpu=use_gpu,
            render_frequency=args.viz_freq,
            enable_rendering=enable_rendering
        )


if __name__ == "__main__":
    main() 