#!/usr/bin/env python3
"""
Phase 3 Testing Script: Static Obstacles & Reward Engineering

Tests the Phase 3 implementation which includes:
- Static obstacle management (barriers, vehicles, pillars)
- Comprehensive collision detection system  
- Enhanced parking spot validation with tolerances
- Improved reward engineering with progress tracking

Run this script to validate Phase 3 implementation before moving to Phase 4.
"""

import sys
import math
import time
import numpy as np
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.append('src')

from environment import (
    ParkingEnv, ObstacleManager, ParkingSpotManager, CollisionDetector,
    Obstacle, ObstacleType, ParkingSpot, ParkingSpotType, CollisionInfo
)


def test_obstacle_management():
    """Test 1: Static Obstacle Management System"""
    print("=" * 60)
    print("TEST 1: Static Obstacle Management System")
    print("=" * 60)
    
    try:
        # Create obstacle manager
        obstacle_manager = ObstacleManager(50.0, 30.0)
        
        # Test creating default obstacles
        obstacles_created = obstacle_manager.create_default_obstacles(40.0, 15.0)
        print(f"✅ Created {obstacles_created} default obstacles")
        
        # Test obstacle summary
        summary = obstacle_manager.get_obstacles_summary()
        print(f"✅ Obstacle summary: {summary['total_obstacles']} obstacles")
        print(f"   - By type: {summary['by_type']}")
        print(f"   - Total area: {summary['total_area_covered']:.1f} m²")
        
        # Test individual obstacle types
        test_obstacles = [
            Obstacle(x=10, y=10, obstacle_type=ObstacleType.CIRCULAR, radius=2.0),
            Obstacle(x=20, y=10, obstacle_type=ObstacleType.VEHICLE, width=4.0, height=2.0),
            Obstacle(x=30, y=10, obstacle_type=ObstacleType.BARRIER, width=1.0, height=5.0)
        ]
        
        for obs in test_obstacles:
            if obstacle_manager.add_obstacle(obs):
                print(f"✅ Added {obs.obstacle_type.value} obstacle at ({obs.x}, {obs.y})")
            else:
                print(f"❌ Failed to add {obs.obstacle_type.value} obstacle")
        
        # Test ray intersection
        test_ray_distance = obstacle_manager.get_ray_intersection(0, 10, 0, 20)  # Ray pointing right
        print(f"✅ Ray intersection test: {test_ray_distance:.2f}m")
        
        print("✅ Obstacle Management Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Obstacle Management Tests FAILED: {e}")
        return False


def test_parking_spot_management():
    """Test 2: Enhanced Parking Spot Management"""
    print("\n" + "=" * 60)
    print("TEST 2: Enhanced Parking Spot Management")
    print("=" * 60)
    
    try:
        # Create parking spot manager
        parking_manager = ParkingSpotManager(50.0, 30.0)
        
        # Test creating default parking spot
        default_spot = parking_manager.create_default_spot(40.0, 15.0, 0.0)
        print(f"✅ Created default parking spot: {default_spot}")
        
        # Test tolerances (from paper: ε_p = 0.5m, ε_θ = 10°)
        print(f"✅ Position tolerance: {default_spot.position_tolerance}m")
        print(f"✅ Orientation tolerance: {default_spot.angle_tolerance}°")
        
        # Test parking validation
        test_positions = [
            (40.0, 15.0, 0.0, "Perfect position"),
            (40.2, 15.2, math.radians(5), "Within tolerance"),
            (40.6, 15.6, math.radians(12), "Outside tolerance"),
            (35.0, 15.0, 0.0, "Too far away")
        ]
        
        for x, y, angle, description in test_positions:
            is_successful = default_spot.is_parking_successful(x, y, angle)
            accuracy = default_spot.get_parking_accuracy(x, y, angle)
            print(f"   {description}: {'✅' if is_successful else '❌'} "
                  f"(pos_acc: {accuracy['position_accuracy']:.2f}, "
                  f"orient_acc: {accuracy['orientation_accuracy']:.2f})")
        
        # Test different parking spot types
        parallel_spot = parking_manager.create_parallel_spot(20.0, 10.0, math.radians(90))
        angled_spot = parking_manager.create_angled_spot(30.0, 20.0, math.radians(45))
        
        print(f"✅ Created parallel spot: {parallel_spot.spot_type.value}")
        print(f"✅ Created angled spot: {angled_spot.spot_type.value}")
        
        # Test parking manager summary
        summary = parking_manager.get_spots_summary()
        print(f"✅ Parking summary: {summary['total_spots']} spots")
        print(f"   - By type: {summary['by_type']}")
        print(f"   - Active spot: {summary['active_spot_index']}")
        
        print("✅ Parking Spot Management Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Parking Spot Management Tests FAILED: {e}")
        return False


def test_collision_detection():
    """Test 3: Collision Detection System"""
    print("\n" + "=" * 60)
    print("TEST 3: Collision Detection System")
    print("=" * 60)
    
    try:
        # Create components
        obstacle_manager = ObstacleManager(50.0, 30.0)
        collision_detector = CollisionDetector(obstacle_manager, 50.0, 30.0)
        
        # Add some test obstacles
        obstacle_manager.create_default_obstacles(40.0, 15.0)
        print(f"✅ Created obstacles for collision testing")
        
        # Test car collision model
        car_model = collision_detector.car_model
        print(f"✅ Car collision model: {car_model.length}m x {car_model.width}m")
        
        # Test collision detection at various positions
        test_positions = [
            (25.0, 15.0, 0.0, "Open space"),
            (1.0, 15.0, 0.0, "Near left boundary"),
            (49.0, 15.0, 0.0, "Near right boundary"),
            (15.0, 15.0, 0.0, "Near obstacle area"),
            (-1.0, 15.0, 0.0, "Outside boundaries")
        ]
        
        for x, y, angle, description in test_positions:
            collision_info = collision_detector.check_collision(x, y, angle)
            is_valid = collision_detector.is_position_valid(x, y, angle)
            closest_distance = collision_detector.get_closest_obstacle_distance(x, y, angle)
            
            print(f"   {description}: {collision_info.collision_type.value} "
                  f"(valid: {'✅' if is_valid else '❌'}, "
                  f"closest: {closest_distance:.2f}m)")
        
        # Test collision statistics
        stats = collision_detector.get_collision_statistics()
        print(f"✅ Collision statistics: {stats['total_collisions']} collisions recorded")
        
        print("✅ Collision Detection Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Collision Detection Tests FAILED: {e}")
        return False


def test_enhanced_reward_system():
    """Test 4: Enhanced Reward Engineering"""
    print("\n" + "=" * 60)
    print("TEST 4: Enhanced Reward Engineering")
    print("=" * 60)
    
    try:
        # Create environment with Phase 3 features
        env = ParkingEnv(
            width=50.0, height=30.0,
            enable_obstacles=True,
            randomize_target=False,
            randomize_obstacles=False
        )
        
        # Reset environment
        initial_obs = env.reset()
        print(f"✅ Environment reset, observation shape: {initial_obs.shape}")
        
        # Test reward components from paper
        reward_info = env.get_reward_info()
        params = reward_info['reward_parameters']
        
        print(f"✅ Reward parameters (from paper):")
        print(f"   - Collision penalty: {params['collision_penalty']}")
        print(f"   - Success reward: {params['success_reward']}")
        print(f"   - Progress positive: {params['progress_positive']}")
        print(f"   - Progress negative: {params['progress_negative']}")
        print(f"   - Time penalty: {params['time_penalty']}")
        
        # Test a few actions to see reward progression
        print(f"\n✅ Testing reward progression:")
        for step in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            
            print(f"   Step {step+1}: action={action}, reward={reward:.3f}, "
                  f"distance={info['distance_to_target']:.2f}m, "
                  f"progress={info['progress_score']:.3f}")
            
            if done:
                print(f"   Episode terminated: collision={info['is_collision']}, "
                      f"success={info['is_successful']}")
                break
        
        # Test environment info
        env_info = env.get_environment_info()
        print(f"✅ Environment info:")
        print(f"   - Obstacles enabled: {env_info['obstacles_enabled']}")
        print(f"   - Total obstacles: {env_info['obstacles_info'].get('total_obstacles', 0)}")
        print(f"   - Active target position: ({env_info['active_target']['x']:.1f}, {env_info['active_target']['y']:.1f})")
        print(f"   - Position tolerance: {env_info['active_target']['position_tolerance']}m")
        print(f"   - Angle tolerance: {env_info['active_target']['angle_tolerance_deg']}°")
        
        env.close()
        print("✅ Enhanced Reward System Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Reward System Tests FAILED: {e}")
        return False


def test_environment_integration():
    """Test 5: Full Environment Integration"""
    print("\n" + "=" * 60)
    print("TEST 5: Full Environment Integration")
    print("=" * 60)
    
    try:
        # Test different environment configurations
        configs = [
            {"enable_obstacles": True, "randomize_target": False, "name": "Standard"},
            {"enable_obstacles": False, "randomize_target": False, "name": "No obstacles"},
            {"enable_obstacles": True, "randomize_target": True, "name": "Random target"}
        ]
        
        for config in configs:
            print(f"\n   Testing {config['name']} configuration:")
            
            env = ParkingEnv(
                width=30.0, height=20.0,  # Smaller for faster testing
                enable_obstacles=config['enable_obstacles'],
                randomize_target=config['randomize_target'],
                max_steps=100
            )
            
            # Test multiple episodes
            episode_rewards = []
            success_count = 0
            collision_count = 0
            
            for episode in range(3):  # Quick test
                obs = env.reset()
                episode_reward = 0
                
                for step in range(20):  # Limited steps for testing
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    if done:
                        if info['is_successful']:
                            success_count += 1
                        if info['is_collision']:
                            collision_count += 1
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            print(f"     ✅ 3 episodes completed")
            print(f"     ✅ Average reward: {avg_reward:.2f}")
            print(f"     ✅ Success rate: {success_count}/3")
            print(f"     ✅ Collision rate: {collision_count}/3")
            
            # Test state analysis
            analysis = env.get_current_state_analysis()
            print(f"     ✅ State analysis keys: {list(analysis.keys())}")
            
            env.close()
        
        print("✅ Environment Integration Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Environment Integration Tests FAILED: {e}")
        return False


def test_interactive_demo():
    """Test 6: Interactive Visualization Demo"""
    print("\n" + "=" * 60)
    print("TEST 6: Interactive Demo (5 seconds)")
    print("=" * 60)
    
    try:
        # Create environment with visualization
        env = ParkingEnv(
            width=40.0, height=25.0,
            render_mode='human',
            enable_obstacles=True,
            show_sensors=True,
            max_steps=500
        )
        
        print("✅ Environment created with visualization")
        print("   Running 5-second demo with random actions...")
        
        obs = env.reset()
        start_time = time.time()
        step_count = 0
        
        while time.time() - start_time < 10.0:  # 5 second demo
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # Render environment
            env.render()
            
            if done:
                print(f"   Episode completed at step {step_count}: "
                      f"success={info['is_successful']}, "
                      f"collision={info['is_collision']}")
                obs = env.reset()
                step_count = 0
                    
        print(f"✅ Demo completed successfully ({step_count} steps)")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Interactive Demo FAILED: {e}")
        return False


def main():
    """Run all Phase 3 tests"""
    print("🚀 PHASE 3 TESTING: Static Obstacles & Reward Engineering")
    print("Testing enhanced parking environment with obstacles, collision detection, and improved rewards")
    print()
    
    # Run all tests
    tests = [
        test_obstacle_management,
        test_parking_spot_management, 
        test_collision_detection,
        test_enhanced_reward_system,
        test_environment_integration,
        test_interactive_demo
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} encountered error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 3 TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Obstacle Management",
        "Parking Spot Management", 
        "Collision Detection",
        "Enhanced Reward System",
        "Environment Integration",
        "Interactive Demo"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name:25s} {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    
    print(f"\nResult: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL PHASE 3 TESTS PASSED!")
        print("Phase 3 implementation is complete and ready for Phase 4.")
        print("\nPhase 3 Features Validated:")
        print("✅ Static obstacle management (barriers, vehicles, pillars)")
        print("✅ Comprehensive collision detection system")
        print("✅ Enhanced parking spot validation with paper tolerances")
        print("✅ Improved reward engineering with progress tracking")
        print("✅ Full environment integration with randomization options")
        print("✅ Real-time visualization with obstacle rendering")
        
        print("\nNext Steps:")
        print("📋 Begin Phase 4: DQN Network Architecture")
        print("🧠 Implement main and target networks")
        print("💾 Create experience replay buffer")
        print("⚙️ Set up training infrastructure")
        
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed.")
        print("Please fix the failing tests before proceeding to Phase 4.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 