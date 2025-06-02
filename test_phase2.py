#!/usr/bin/env python3
"""
Phase 2 Testing Script
Tests the action space, sensor system, and reward function integration.

Based on Phase 2 objectives from the roadmap:
- Implement discrete action space (7 actions as per paper)
- Add distance sensors for obstacle detection
- Create basic reward function structure  
- Test agent-environment interaction loop
"""

import sys
import os
import numpy as np
import time
import pygame

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment import (
    ParkingEnv, ActionSpace, ActionType, SensorArray, 
    RewardFunction, RewardType
)


class Phase2Tester:
    """Comprehensive testing for Phase 2 implementation."""
    
    def __init__(self):
        """Initialize the tester."""
        self.results = {}
        self.env = None
        
    def run_all_tests(self) -> bool:
        """Run all Phase 2 tests."""
        print("🚀 Starting Phase 2 Testing...")
        print("=" * 60)
        
        tests = [
            ("Action Space Tests", self.test_action_space),
            ("Sensor System Tests", self.test_sensor_system), 
            ("Reward Function Tests", self.test_reward_function),
            ("Environment Integration Tests", self.test_environment_integration),
            ("Interactive Demo", self.test_interactive_demo)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                self.results[test_name] = result
                
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    all_passed = False
                    
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                self.results[test_name] = False
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 PHASE 2 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:30s} {status}")
        
        if all_passed:
            print(f"\n🎉 ALL TESTS PASSED! Phase 2 implementation is complete.")
            print("✨ Ready to proceed to Phase 3: Static Obstacles & Reward Engineering")
        else:
            print(f"\n⚠️  Some tests failed. Please review and fix issues.")
        
        return all_passed
    
    def test_action_space(self) -> bool:
        """Test the action space implementation."""
        print("Testing ActionSpace class...")
        
        try:
            # Test action space creation
            action_space = ActionSpace()
            
            # Test basic properties
            assert action_space.n_actions == 7, f"Expected 7 actions, got {action_space.n_actions}"
            print(f"  ✓ Action space has {action_space.n_actions} actions")
            
            # Test action parameters
            expected_actions = {
                0: (-0.6, 0.0),   # Hold/Brake
                1: (+0.6, 0.0),   # Throttle forward
                2: (-0.6, 0.0),   # Reverse back
                3: (+0.6, +8.0),  # Left forward
                4: (+0.6, -8.0),  # Right forward
                5: (-0.6, +8.0),  # Left reverse
                6: (-0.6, -8.0),  # Right reverse
            }
            
            for action_id, (expected_vel, expected_steer) in expected_actions.items():
                vel_change, steer_change = action_space.get_action_params(action_id)
                assert abs(vel_change - expected_vel) < 0.001, f"Action {action_id} velocity mismatch"
                assert abs(steer_change - expected_steer) < 0.001, f"Action {action_id} steering mismatch"
            
            print(f"  ✓ All action parameters match paper specifications")
            
            # Test action descriptions
            for action_id in range(7):
                desc = action_space.get_action_description(action_id)
                assert isinstance(desc, str) and len(desc) > 0, f"Invalid description for action {action_id}"
            
            print(f"  ✓ Action descriptions available")
            
            # Test action validation
            assert action_space.is_valid_action(0), "Action 0 should be valid"
            assert action_space.is_valid_action(6), "Action 6 should be valid"
            assert not action_space.is_valid_action(7), "Action 7 should be invalid"
            assert not action_space.is_valid_action(-1), "Action -1 should be invalid"
            
            print(f"  ✓ Action validation working correctly")
            
            # Test action effects matrix
            effects = action_space.get_action_effects_matrix()
            assert effects.shape == (7, 2), f"Expected (7,2) effects matrix, got {effects.shape}"
            
            print(f"  ✓ Action effects matrix generated")
            
            # Print action space summary
            print(f"\n{action_space}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Action space test failed: {e}")
            return False
    
    def test_sensor_system(self) -> bool:
        """Test the sensor system implementation."""
        print("Testing SensorArray class...")
        
        try:
            # Test sensor array creation
            sensor_array = SensorArray(max_range=20.0)
            
            # Test basic properties
            assert len(sensor_array.sensors) == 8, f"Expected 8 sensors, got {len(sensor_array.sensors)}"
            assert sensor_array.max_range == 20.0, f"Expected max_range 20.0, got {sensor_array.max_range}"
            
            print(f"  ✓ Sensor array has {len(sensor_array.sensors)} sensors")
            
            # Test sensor angles (should be every 45 degrees)
            for i, sensor in enumerate(sensor_array.sensors):
                expected_angle = i * (np.pi / 4)
                assert abs(sensor.relative_angle - expected_angle) < 0.001, f"Sensor {i} angle mismatch"
            
            print(f"  ✓ Sensor angles correctly positioned (45° intervals)")
            
            # Test sensor readings with mock environment
            environment_bounds = (0.0, 0.0, 50.0, 30.0)  # 50x30 meter environment
            car_x, car_y, car_theta = 25.0, 15.0, 0.0  # Center of environment
            
            readings = sensor_array.get_all_readings(
                car_x, car_y, car_theta, environment_bounds
            )
            
            assert len(readings) == 8, f"Expected 8 readings, got {len(readings)}"
            assert all(0 <= r <= 20.0 for r in readings), "All readings should be within max_range"
            
            print(f"  ✓ Sensor readings generated: {[f'{r:.1f}' for r in readings]}")
            
            # Test specific sensor groups
            front_sensors = sensor_array.get_front_sensors()
            rear_sensors = sensor_array.get_rear_sensors()
            side_sensors = sensor_array.get_side_sensors()
            
            assert len(front_sensors) == 3, "Front sensors should return 3 values"
            assert len(rear_sensors) == 3, "Rear sensors should return 3 values"
            assert len(side_sensors) == 2, "Side sensors should return 2 values"
            
            print(f"  ✓ Sensor grouping: front={front_sensors}, rear={rear_sensors}, sides={side_sensors}")
            
            # Test collision risk detection
            risk_analysis = sensor_array.detect_collision_risk(threshold_distance=5.0)
            assert 'has_risk' in risk_analysis, "Risk analysis should include 'has_risk'"
            assert 'min_distance' in risk_analysis, "Risk analysis should include 'min_distance'"
            
            print(f"  ✓ Collision risk detection: {risk_analysis}")
            
            # Test sensor normalization
            normalized = sensor_array.normalize_readings(readings)
            assert len(normalized) == 8, "Normalized readings should have 8 values"
            assert all(0.0 <= r <= 1.0 for r in normalized), "Normalized readings should be in [0,1]"
            
            print(f"  ✓ Sensor normalization working")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Sensor system test failed: {e}")
            return False
    
    def test_reward_function(self) -> bool:
        """Test the reward function implementation."""
        print("Testing RewardFunction class...")
        
        try:
            # Test reward function creation
            reward_func = RewardFunction()
            
            # Test basic properties (from paper)
            assert reward_func.collision_penalty == -100.0, "Collision penalty should be -100"
            assert reward_func.success_reward == +100.0, "Success reward should be +100"
            assert reward_func.progress_reward_positive == +1.0, "Progress reward should be +1"
            assert reward_func.progress_reward_negative == -0.5, "Progress penalty should be -0.5"
            assert reward_func.time_penalty == -0.1, "Time penalty should be -0.1"
            assert reward_func.position_tolerance == 0.5, "Position tolerance should be 0.5m"
            
            print(f"  ✓ Reward parameters match paper specifications")
            
            # Test collision reward
            collision_reward = reward_func.calculate_reward(
                car_x=10.0, car_y=10.0, car_theta=0.0, car_velocity=1.0,
                target_x=20.0, target_y=20.0, target_theta=0.0,
                is_collision=True, is_out_of_bounds=False,
                sensor_readings=[10.0] * 8, timestep=1
            )
            
            assert collision_reward['total_reward'] == -100.0, "Collision should give -100 reward"
            assert RewardType.COLLISION.value in collision_reward['components'], "Should include collision component"
            
            print(f"  ✓ Collision penalty: {collision_reward['total_reward']}")
            
            # Test success reward
            success_reward = reward_func.calculate_reward(
                car_x=20.0, car_y=20.0, car_theta=0.0, car_velocity=0.1,
                target_x=20.0, target_y=20.0, target_theta=0.0,  # Same position
                is_collision=False, is_out_of_bounds=False,
                sensor_readings=[15.0] * 8, timestep=1
            )
            
            assert success_reward['is_successful'], "Should be successful parking"
            assert success_reward['total_reward'] == 100.0, "Success should give +100 reward"
            
            print(f"  ✓ Success reward: {success_reward['total_reward']}")
            
            # Test time penalty (normal case)
            normal_reward = reward_func.calculate_reward(
                car_x=10.0, car_y=10.0, car_theta=0.0, car_velocity=1.0,
                target_x=15.0, target_y=15.0, target_theta=0.0,
                is_collision=False, is_out_of_bounds=False,
                sensor_readings=[10.0] * 8, timestep=1
            )
            
            assert RewardType.TIME.value in normal_reward['components'], "Should include time penalty"
            assert normal_reward['components'][RewardType.TIME.value] == -0.1, "Time penalty should be -0.1"
            
            print(f"  ✓ Time penalty applied: {normal_reward['components'][RewardType.TIME.value]}")
            
            # Test successful parking tolerance
            assert reward_func._is_successful_parking(
                20.0, 20.0, 0.0,  # Car position/orientation
                20.0, 20.0, 0.0   # Target position/orientation (same)
            ), "Should detect successful parking at exact target"
            
            assert reward_func._is_successful_parking(
                20.3, 20.3, np.radians(5),  # Close to target
                20.0, 20.0, 0.0
            ), "Should detect successful parking within tolerance"
            
            assert not reward_func._is_successful_parking(
                21.0, 21.0, np.radians(15),  # Too far from target
                20.0, 20.0, 0.0
            ), "Should not detect parking outside tolerance"
            
            print(f"  ✓ Parking success detection with tolerances working")
            
            print(f"\n{reward_func}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Reward function test failed: {e}")
            return False
    
    def test_environment_integration(self) -> bool:
        """Test the integrated environment with all Phase 2 components."""
        print("Testing ParkingEnv integration...")
        
        try:
            # Create environment
            env = ParkingEnv(
                width=50.0, height=30.0, 
                sensor_max_range=20.0, 
                show_sensors=True
            )
            
            # Test environment properties
            assert env.action_space.n == 7, f"Expected 7 actions, got {env.action_space.n}"
            assert env.observation_space.shape == (12,), f"Expected state dimension 12, got {env.observation_space.shape}"
            
            print(f"  ✓ Environment created with {env.action_space.n} actions, state dim {env.observation_space.shape}")
            
            # Test environment reset
            initial_obs = env.reset()
            assert initial_obs.shape == (12,), f"Initial observation should have 12 dimensions"
            assert len(initial_obs) == 12, "State should be [x, y, θ, v, d1...d8]"
            
            print(f"  ✓ Environment reset, initial state: {initial_obs}")
            
            # Test all actions
            action_tests = env.test_all_actions()
            assert len(action_tests) == 7, "Should have 7 action descriptions"
            
            for action_id, description in action_tests.items():
                print(f"    Action {action_id}: {description}")
            
            print(f"  ✓ All actions accessible")
            
            # Test environment step with each action
            total_reward = 0
            for action in range(7):
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                # Validate step return values
                assert obs.shape == (12,), f"Observation shape should be (12,), got {obs.shape}"
                assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
                assert isinstance(done, bool), f"Done should be boolean, got {type(done)}"
                assert isinstance(info, dict), f"Info should be dict, got {type(info)}"
                
                # Check info contents
                required_info = ['step', 'total_reward', 'reward_components', 'distance_to_target', 
                               'is_collision', 'is_successful', 'action_description', 'sensor_readings']
                for key in required_info:
                    assert key in info, f"Info should contain '{key}'"
                
                if done:
                    env.reset()
                    break
            
            print(f"  ✓ Environment stepping working, total reward: {total_reward:.2f}")
            
            # Test environment info methods
            action_info = env.get_action_space_info()
            sensor_info = env.get_sensor_info()
            reward_info = env.get_reward_info()
            state_analysis = env.get_current_state_analysis()
            
            assert 'n_actions' in action_info, "Action info should include n_actions"
            assert 'n_sensors' in sensor_info, "Sensor info should include n_sensors"
            assert 'collision_penalty' in reward_info, "Reward info should include collision_penalty"
            assert 'car_state' in state_analysis, "State analysis should include car_state"
            
            print(f"  ✓ Environment info methods working")
            print(f"    Actions: {action_info['n_actions']}, Sensors: {sensor_info['n_sensors']}")
            
            self.env = env  # Save for interactive demo
            
            return True
            
        except Exception as e:
            print(f"  ❌ Environment integration test failed: {e}")
            return False
    
    def test_interactive_demo(self) -> bool:
        """Run an interactive demo of the environment."""
        print("Running interactive demo...")
        
        if self.env is None:
            print("  ⚠️  Environment not available for demo")
            return False
        
        try:
            # Initialize pygame for rendering
            pygame.init()
            
            print("  🎮 Starting interactive demo...")
            print("  📋 Controls:")
            print("    0: Hold/Brake     1: Forward      2: Reverse")
            print("    3: Left+Forward   4: Right+Forward")
            print("    5: Left+Reverse   6: Right+Reverse")
            print("    ESC: Exit demo")
            
            # Demo parameters
            demo_steps = 100
            action_duration = 5  # Steps per action
            
            obs = self.env.reset()
            step_count = 0
            action_step = 0
            current_action = 1  # Start with forward throttle
            
            clock = pygame.time.Clock()
            
            while step_count < demo_steps:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print("  👋 Demo ended by user")
                        return True
                    elif event.type == pygame.KEYDOWN:
                        if pygame.K_0 <= event.key <= pygame.K_6:
                            current_action = event.key - pygame.K_0
                            action_step = 0
                            print(f"  🎮 Action changed to {current_action}: {self.env.action_space_manager.get_action_description(current_action)}")
                
                # Auto change action for demo
                if action_step >= action_duration:
                    current_action = (current_action + 1) % 7
                    action_step = 0
                    print(f"  🤖 Auto action {current_action}: {self.env.action_space_manager.get_action_description(current_action)}")
                
                # Step environment
                obs, reward, done, info = self.env.step(current_action)
                
                # Render
                self.env.render()
                
                # Print status
                if step_count % 10 == 0:
                    state_analysis = self.env.get_current_state_analysis()
                    car_state = state_analysis['car_state']
                    print(f"  📊 Step {step_count}: Pos=({car_state['position'][0]:.1f}, {car_state['position'][1]:.1f}), "
                          f"θ={car_state['orientation_deg']:.1f}°, Reward={reward:.2f}")
                
                # Check termination
                if done:
                    if info['is_successful']:
                        print(f"  🎉 SUCCESS! Car parked successfully at step {step_count}")
                    elif info['is_collision']:
                        print(f"  💥 COLLISION! Episode ended at step {step_count}")
                    else:
                        print(f"  ⏱️  Episode ended (timeout/boundary) at step {step_count}")
                    
                    obs = self.env.reset()
                    action_step = 0
                
                step_count += 1
                action_step += 1
                clock.tick(10)  # 10 FPS
            
            print(f"  ✅ Interactive demo completed successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ Interactive demo failed: {e}")
            return False
        finally:
            if self.env:
                self.env.close()


def main():
    """Main test execution."""
    print("🚗 Autonomous Parking DQN - Phase 2 Testing")
    print("Phase 2: Action Space & Basic Interaction")
    print("=" * 60)
    
    tester = Phase2Tester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Phase 2 is COMPLETE and ready for Phase 3!")
        print("📋 Next steps:")
        print("  1. Review test results and implementation")
        print("  2. Update README.md to mark Phase 2 as complete")
        print("  3. Begin Phase 3: Static Obstacles & Reward Engineering")
    else:
        print("\n⚠️  Please fix failing tests before proceeding to Phase 3")
    
    return success


if __name__ == "__main__":
    main() 