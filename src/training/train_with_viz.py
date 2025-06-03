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


def progressive_training_with_viz(start_stage: int = 0, use_gpu: bool = True):
    """
    Run progressive training with visualization.
    
    Args:
        start_stage: Which stage to start from (0=simple, 1=obstacles, 2=full)
        use_gpu: Whether to use GPU acceleration
    """
    
    # Define progressive stages
    stages = [
        ("progressive_simple", "🟢 Stage 1: Simple Environment (No Obstacles)"),
        ("progressive_obstacles", "🟡 Stage 2: With Obstacles"),
        ("progressive_full", "🔴 Stage 3: Full Complexity")
    ]
    
    print(f"🚀 PROGRESSIVE TRAINING WITH VISUALIZATION")
    print(f"Starting from stage: {start_stage + 1}")
    print(f"GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
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
        print(f"   🎮 Visualization: ON (every episode)")
        
        try:
            # Train with visualization - ALWAYS show simulation window
            print(f"\n🚀 Starting training with live visualization...")
            print(f"   Watch the pygame window to see agent behavior!")
            
            results = trainer.train(
                episodes=config.total_episodes,
                render_during_training=True,  # Always show visualization
                render_frequency=1,           # Render every episode
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


def single_stage_training_with_viz(stage: str, use_gpu: bool = True):
    """Train a single stage with visualization."""
    
    print(f"🎯 Single Stage Training with Visualization: {stage}")
    print(f"GPU Acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    
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
    print(f"   🎮 Visualization: ON (every episode)")
    
    try:
        # Start training with visualization
        print(f"\n🚀 Starting training with live visualization...")
        print(f"   Watch the pygame window to see agent behavior!")
        
        results = trainer.train(
            episodes=config.total_episodes,
            render_during_training=True,  # Always show visualization
            render_frequency=1,           # Render every episode
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


def main():
    """Main training execution with argument parsing."""
    parser = argparse.ArgumentParser(description='Progressive DQN Training with Visualization')
    parser.add_argument('--mode', choices=['progressive', 'single'], default='progressive',
                       help='Training mode: progressive (3 stages) or single stage')
    parser.add_argument('--stage', choices=['progressive_simple', 'progressive_obstacles', 'progressive_full', 'debug_viz'], 
                       default='progressive_simple', help='Single stage to train')
    parser.add_argument('--start-stage', type=int, default=0, choices=[0, 1, 2],
                       help='Which stage to start progressive training from (0=simple, 1=obstacles, 2=full)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage (disable GPU)')
    
    args = parser.parse_args()
    
    use_gpu = not args.cpu
    
    print(f"🎮 Phase 7: Progressive DQN Training with Visualization")
    print(f"Mode: {args.mode}")
    print(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    
    if args.mode == 'progressive':
        # Progressive training
        progressive_training_with_viz(
            start_stage=args.start_stage,
            use_gpu=use_gpu
        )
    else:
        # Single stage training
        single_stage_training_with_viz(args.stage, use_gpu=use_gpu)


if __name__ == "__main__":
    main() 