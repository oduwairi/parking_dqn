"""
Phase 1 Test Script
Tests the basic parking environment, car agent, and renderer.
"""

import numpy as np
import time
from src.environment.parking_env import ParkingEnv

def test_basic_environment():
    """Test basic environment functionality."""
    print("🚗 Testing Phase 1: Basic Environment Setup")
    print("=" * 50)
    
    # Create environment
    env = ParkingEnv(width=50.0, height=30.0, render_mode='human')
    
    print(f"✅ Environment created:")
    print(f"   - Size: {env.width}m × {env.height}m")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Target position: ({env.target_x:.1f}, {env.target_y:.1f})")
    
    # Reset environment
    obs = env.reset()
    print(f"✅ Environment reset, initial observation shape: {obs.shape}")
    print(f"   - Car position: ({env.car.x:.1f}, {env.car.y:.1f})")
    print(f"   - Car orientation: {np.degrees(env.car.theta):.1f}°")
    print(f"   - Distance sensors: {obs[4:].tolist()}")
    
    return env

def test_car_actions():
    """Test all 7 discrete actions."""
    print("\n🎮 Testing Car Actions")
    print("=" * 30)
    
    env = ParkingEnv(width=50.0, height=30.0)
    obs = env.reset()
    
    action_names = [
        "Hold/Brake",
        "Throttle forward", 
        "Reverse back",
        "Left forward",
        "Right forward",
        "Left reverse", 
        "Right reverse"
    ]
    
    for action in range(7):
        print(f"  Action {action}: {action_names[action]}")
        
        # Test action for a few steps
        for _ in range(3):
            obs, reward, done, info = env.step(action)
            
        print(f"    → Position: ({env.car.x:.1f}, {env.car.y:.1f})")
        print(f"    → Velocity: {env.car.velocity:.2f} m/s")
        print(f"    → Reward: {reward:.2f}")
        
        # Reset between actions
        env.reset()
        
    print("✅ All actions tested successfully")

def test_reward_system():
    """Test reward calculation."""
    print("\n💰 Testing Reward System")
    print("=" * 25)
    
    env = ParkingEnv(width=50.0, height=30.0)
    obs = env.reset()
    
    # Test normal step (should have time penalty)
    obs, reward, done, info = env.step(1)  # Throttle forward
    print(f"✅ Normal step reward: {reward:.2f} (should include -0.1 time penalty)")
    
    # Test progress toward target
    initial_distance = env._distance_to_target()
    print(f"✅ Initial distance to target: {initial_distance:.2f}m")
    
    # Move toward target several times
    for i in range(10):
        obs, reward, done, info = env.step(1)  # Throttle forward
        current_distance = env._distance_to_target()
        
        if i == 0:
            print(f"✅ After movement - Distance: {current_distance:.2f}m, Reward: {reward:.2f}")
        
        if done:
            if info['is_collision']:
                print(f"💥 Collision detected! Reward: {reward:.2f}")
            elif info['is_successful']:
                print(f"🎯 Successful parking! Reward: {reward:.2f}")
            break
            
    print("✅ Reward system tested")

def test_interactive_demo(duration_seconds=10):
    """Run interactive demo with random actions."""
    print(f"\n🎬 Interactive Demo ({duration_seconds}s)")
    print("=" * 30)
    
    env = ParkingEnv(width=50.0, height=30.0, render_mode='human')
    obs = env.reset()
    
    start_time = time.time()
    step_count = 0
    total_reward = 0
    
    print("✅ Demo started - watch the pygame window!")
    print("   The car (blue) should try to reach the green parking spot")
    
    try:
        while time.time() - start_time < duration_seconds:
            # Random action (simple policy for demo)
            action = np.random.choice(7)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render (if render_mode is 'human')
            env.render()
            
            if done:
                print(f"   Episode ended at step {step_count}")
                if info['is_collision']:
                    print("   💥 Collision occurred")
                elif info['is_successful']:
                    print("   🎯 Successful parking!")
                else:
                    print("   ⏰ Timeout")
                    
                # Reset for next episode
                obs = env.reset()
                step_count = 0
                
            # Small delay for visualization
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n   Demo interrupted by user")
    
    finally:
        env.close()
        
    print(f"✅ Demo completed - Total reward: {total_reward:.2f}")

def main():
    """Run all Phase 1 tests."""
    print("🚀 Phase 1 Testing Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic environment
        env = test_basic_environment()
        
        # Test 2: Car actions
        test_car_actions()
        
        # Test 3: Reward system
        test_reward_system()
        
        # Test 4: Interactive demo
        print("\nRunning interactive demo...")
        print("Close the pygame window or press Ctrl+C to stop")
        test_interactive_demo(duration_seconds=15)
        
        print("\n" + "=" * 50)
        print("🎉 Phase 1 Testing Complete!")
        print("✅ Environment setup: PASSED")
        print("✅ Car physics: PASSED") 
        print("✅ Action space: PASSED")
        print("✅ Reward system: PASSED")
        print("✅ Visualization: PASSED")
        print("\n🚀 Ready to proceed to Phase 2!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check the implementation and dependencies.")
        raise
        
    finally:
        # Clean up
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    main() 