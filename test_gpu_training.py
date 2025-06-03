#!/usr/bin/env python3
"""
Quick GPU Test for Progressive DQN Training
Tests if CUDA is working and runs a short training session with visualization.
"""

import os
import sys
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_gpu_availability():
    """Test if GPU is available and working."""
    print("🔍 GPU Availability Test")
    print("=" * 50)
    
    try:
        import torch
        print(f"✅ PyTorch Version: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"✅ GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Test tensor operation on GPU
            device = torch.device("cuda")
            x = torch.randn(1000, 1000).to(device)
            y = torch.mm(x, x)
            print(f"✅ GPU Tensor Test: PASSED")
            
            return True
        else:
            print("❌ CUDA not available - will use CPU")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False


def test_quick_training(use_gpu=True):
    """Run a quick training test with visualization."""
    print(f"\n🎮 Quick Training Test ({'GPU' if use_gpu else 'CPU'})")
    print("=" * 50)
    
    try:
        from src.training.train_with_viz import single_stage_training_with_viz
        
        # Run a very short training session
        print("🚀 Starting 25-episode training test with visualization...")
        print("   Watch the pygame window!")
        
        # Override config for very short test
        from src.training.config import get_config
        config = get_config('debug_viz')
        config.total_episodes = 25
        config.max_steps_per_episode = 50
        config.use_gpu = use_gpu
        config.log_frequency = 5
        config.evaluation_frequency = 10
        
        # Create temporary trainer
        from src.training.trainer import DQNTrainer
        experiment_name = f"gpu_test_{int(time.time())}"
        trainer = DQNTrainer(config=config, experiment_name=experiment_name)
        
        start_time = time.time()
        results = trainer.train(
            episodes=25,
            render_during_training=True,
            render_frequency=1,
            verbose=True
        )
        training_time = time.time() - start_time
        
        print(f"\n📊 Test Results:")
        print(f"   ⏱️ Training Time: {training_time:.1f} seconds")
        print(f"   📈 Episodes/sec: {25/training_time:.1f}")
        print(f"   🎯 Device Used: {'GPU' if config.use_gpu and torch.cuda.is_available() else 'CPU'}")
        
        final_eval = results.get('final_evaluation', {})
        if final_eval:
            print(f"   🏆 Final Success Rate: {final_eval.get('success_rate', 0):.1%}")
            print(f"   💥 Final Collision Rate: {final_eval.get('collision_rate', 0):.1%}")
            print(f"   🎁 Final Avg Reward: {final_eval.get('avg_reward', 0):+.2f}")
        
        print(f"✅ Training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution."""
    print("🧪 GPU Training Test Suite")
    print("=" * 60)
    
    # Test 1: GPU availability
    gpu_available = test_gpu_availability()
    
    # Test 2: Quick training
    if gpu_available:
        print("\n🎯 Testing GPU-accelerated training...")
        success = test_quick_training(use_gpu=True)
    else:
        print("\n🎯 Testing CPU training (GPU not available)...")
        success = test_quick_training(use_gpu=False)
    
    # Summary
    print(f"\n🏁 Test Summary")
    print("=" * 30)
    if gpu_available and success:
        print("✅ GPU acceleration: READY")
        print("✅ Progressive training: READY")
        print("✅ Visualization: WORKING")
        print("\n🚀 Ready for full progressive training!")
        print("   Run: python src/training/train_with_viz.py --mode progressive")
    elif success:
        print("✅ CPU training: WORKING")
        print("✅ Visualization: WORKING")
        print("⚠️ GPU acceleration: NOT AVAILABLE")
        print("\n🔧 Install CUDA PyTorch for GPU acceleration:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("❌ Training system: ISSUES DETECTED")
        print("🔧 Check installation and try again")


if __name__ == "__main__":
    main() 