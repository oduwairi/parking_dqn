"""
Training script with visualization for debugging agent behavior.
This allows you to watch the agent learn in real-time.
"""

import argparse
import time
import os
import sys
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from trainer import DQNTrainer
from config import get_config


def train_with_visualization():
    """Train DQN agent with real-time visualization."""
    
    # Use debug configuration for visualization
    config = get_config('debug_viz')
    
    # Create trainer
    trainer = DQNTrainer(config=config)
    
    print("🎮 Starting DQN Training with Visualization")
    print("   - Episodes: 50 (for debugging)")
    print("   - Visualization: ON")
    print("   - Sensors shown: YES")
    print("   - Press Ctrl+C to stop training\n")
    
    try:
        # Train with visualization
        trainer.train(
            episodes=50,
            render_during_training=True,  # This will show the pygame window
            render_frequency=1,           # Render every episode
            verbose=True
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
    finally:
        print("🏁 Training session ended")


if __name__ == "__main__":
    train_with_visualization() 