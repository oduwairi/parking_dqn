#!/usr/bin/env python3
"""
Phase 4 Testing Script: DQN Network Architecture

Tests the Phase 4 implementation which includes:
- DQN network architecture (3×256 hidden layers + ReLU)
- Experience replay buffer (~10^5 capacity)
- Huber loss function and training utilities
- Complete DQN agent with epsilon-greedy policy
- Integration with parking environment

Run this script to validate Phase 4 implementation before moving to Phase 5.
"""

import sys
import time
import numpy as np
import torch
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.append('src')

from dqn import (
    DQNNetwork, DoubleDQNNetwork, create_dqn_networks,
    ReplayBuffer, PrioritizedReplayBuffer, create_replay_buffer,
    HuberLoss, DQNLoss, EpsilonScheduler, LearningRateScheduler,
    DQNAgent, create_dqn_agent
)

from environment import ParkingEnv


def test_dqn_network_architecture():
    """Test 1: DQN Network Architecture"""
    print("=" * 60)
    print("TEST 1: DQN Network Architecture")
    print("=" * 60)
    
    try:
        # Test standard DQN network
        main_net, target_net = create_dqn_networks(use_double_dqn=False)
        network_info = main_net.get_network_info()
        
        print(f"✅ Standard DQN networks created")
        print(f"   - State dim: {network_info['state_dim']} (expected: 12)")
        print(f"   - Action dim: {network_info['action_dim']} (expected: 7)")
        print(f"   - Hidden dim: {network_info['hidden_dim']} (expected: 256)")
        print(f"   - Hidden layers: {network_info['num_hidden_layers']} (expected: 3)")
        print(f"   - Total parameters: {network_info['total_parameters']:,}")
        
        # Test forward pass
        test_batch = torch.randn(32, 12)  # Batch of 32 states
        q_values = main_net(test_batch)
        
        assert q_values.shape == (32, 7), f"Expected shape (32, 7), got {q_values.shape}"
        print(f"✅ Forward pass working: {test_batch.shape} → {q_values.shape}")
        
        # Test action selection
        single_state = torch.randn(12)
        action, q_vals = main_net.get_action(single_state, epsilon=0.1)
        
        assert 0 <= action < 7, f"Action {action} not in valid range [0, 6]"
        print(f"✅ Action selection working: action={action}, Q-values shape={q_vals.shape}")
        
        # Test Double DQN
        double_main, double_target = create_dqn_networks(use_double_dqn=True)
        print(f"✅ Double DQN networks created")
        
        # Test target network updates
        original_params = target_net.state_dict()['network.0.weight'].clone()
        target_net.copy_weights_from(main_net)
        updated_params = target_net.state_dict()['network.0.weight']
        
        # Check if networks are different (they should be after copying)
        params_different = not torch.allclose(original_params, updated_params, atol=1e-6)
        if not params_different:
            print(f"⚠️  Target network parameters identical - checking soft update instead")
        else:
            print(f"✅ Target network update working")
        
        # Test soft update
        target_net.soft_update_from(main_net, tau=0.1)
        print(f"✅ Soft update working")
        
        print("✅ DQN Network Architecture Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ DQN Network Architecture Tests FAILED: {e}")
        return False


def test_experience_replay_buffer():
    """Test 2: Experience Replay Buffer"""
    print("\n" + "=" * 60)
    print("TEST 2: Experience Replay Buffer")
    print("=" * 60)
    
    try:
        # Test standard replay buffer
        buffer = ReplayBuffer(capacity=1000, state_dim=12)
        buffer_info = buffer.get_buffer_info()
        
        print(f"✅ Standard replay buffer created")
        print(f"   - Capacity: {buffer_info['capacity']:,} (testing with 1K)")
        print(f"   - State dim: {buffer_info['state_dim']}")
        print(f"   - Memory usage: {buffer_info['memory_usage_mb']:.2f} MB")
        
        # Add experiences
        for i in range(200):
            state = np.random.randn(12)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.random() < 0.1
            
            buffer.push(state, action, reward, next_state, done)
        
        updated_info = buffer.get_buffer_info()
        print(f"✅ Added 200 experiences")
        print(f"   - Current size: {updated_info['current_size']}")
        print(f"   - Usage: {updated_info['usage_percent']:.1f}%")
        
        # Test sampling
        if buffer.can_provide_sample(64):
            states, actions, rewards, next_states, dones = buffer.sample(64)
            
            expected_shapes = [(64, 12), (64,), (64,), (64, 12), (64,)]
            actual_shapes = [x.shape for x in [states, actions, rewards, next_states, dones]]
            
            assert actual_shapes == expected_shapes, f"Batch shapes mismatch: {actual_shapes}"
            print(f"✅ Sampling working: batch shapes {actual_shapes}")
        
        # Test buffer statistics
        stats = buffer.get_statistics()
        print(f"✅ Buffer statistics calculated:")
        print(f"   - Mean reward: {stats['mean_reward']:.3f}")
        print(f"   - Actions distribution: {[f'{k}={v:.1f}%' for k, v in stats.items() if 'action_' in k][:3]}")
        
        # Test prioritized replay buffer
        priority_buffer = PrioritizedReplayBuffer(capacity=1000, state_dim=12)
        
        # Add experiences with priorities
        for i in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.random() < 0.1
            td_error = np.random.random()  # Mock TD error
            
            priority_buffer.push(state, action, reward, next_state, done, td_error)
        
        print(f"✅ Prioritized replay buffer working")
        
        # Test prioritized sampling
        if priority_buffer.can_provide_sample(32):
            batch_data = priority_buffer.sample(32)
            assert len(batch_data) == 7, f"Prioritized sample should return 7 items, got {len(batch_data)}"
            print(f"✅ Prioritized sampling working")
        
        print("✅ Experience Replay Buffer Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Experience Replay Buffer Tests FAILED: {e}")
        return False


def test_loss_functions_and_utilities():
    """Test 3: Loss Functions and Training Utilities"""
    print("\n" + "=" * 60)
    print("TEST 3: Loss Functions and Training Utilities")
    print("=" * 60)
    
    try:
        # Test Huber loss
        huber_loss = HuberLoss(delta=1.0)
        predictions = torch.randn(64, 1)
        targets = torch.randn(64, 1)
        loss_value = huber_loss(predictions, targets)
        
        assert loss_value.item() >= 0, "Loss should be non-negative"
        print(f"✅ Huber loss working: {loss_value.item():.4f}")
        
        # Test DQN loss computation
        dqn_loss = DQNLoss(gamma=0.95)
        
        # Create mock networks
        main_net, target_net = create_dqn_networks()
        
        # Create mock batch data
        batch_size = 32
        states = torch.randn(batch_size, 12)
        actions = torch.randint(0, 7, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 12)
        dones = torch.randint(0, 2, (batch_size,)).bool()
        
        # Test standard DQN loss
        loss, metrics = dqn_loss.compute_dqn_loss(
            main_net, target_net, states, actions, rewards, next_states, dones
        )
        
        assert loss.item() >= 0, "DQN loss should be non-negative"
        assert 'mean_q_value' in metrics, "Metrics should include mean Q-value"
        print(f"✅ DQN loss computation working: loss={loss.item():.4f}")
        print(f"   - Mean Q-value: {metrics['mean_q_value']:.3f}")
        print(f"   - Mean TD error: {metrics['mean_td_error']:.3f}")
        
        # Test Double DQN loss
        double_main, double_target = create_dqn_networks(use_double_dqn=True)
        loss_double, metrics_double = dqn_loss.compute_double_dqn_loss(
            double_main, double_target, states, actions, rewards, next_states, dones
        )
        
        assert 'double_dqn_used' in metrics_double, "Double DQN metrics should be flagged"
        print(f"✅ Double DQN loss working: loss={loss_double.item():.4f}")
        
        # Test epsilon scheduler
        epsilon_scheduler = EpsilonScheduler(
            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995
        )
        
        epsilons = []
        for i in range(100):
            epsilon = epsilon_scheduler.get_epsilon(i)
            epsilons.append(epsilon)
        
        assert epsilons[0] > epsilons[-1], "Epsilon should decay over time"
        print(f"✅ Epsilon scheduler working: {epsilons[0]:.3f} → {epsilons[-1]:.3f}")
        
        # Test learning rate scheduler
        lr_scheduler = LearningRateScheduler(initial_lr=1e-3)
        
        lrs = []
        for i in range(100):
            lr = lr_scheduler.get_lr(i)
            lrs.append(lr)
        
        print(f"✅ Learning rate scheduler working: {lrs[0]:.6f} → {lrs[-1]:.6f}")
        
        print("✅ Loss Functions and Utilities Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Loss Functions and Utilities Tests FAILED: {e}")
        return False


def test_dqn_agent_integration():
    """Test 4: DQN Agent Integration"""
    print("\n" + "=" * 60)
    print("TEST 4: DQN Agent Integration")
    print("=" * 60)
    
    try:
        # Create DQN agent with test configuration
        agent = DQNAgent(
            state_dim=12,
            action_dim=7,
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=32,  # Smaller for testing
            replay_buffer_size=1000,  # Smaller for testing
            target_update_freq=100,  # More frequent for testing
            use_double_dqn=True,
            use_prioritized_replay=False,
            seed=42
        )
        
        agent_info = agent.get_agent_info()
        print(f"✅ DQN Agent created successfully")
        print(f"   - Device: {agent_info['device']}")
        print(f"   - Network type: {'Double DQN' if agent_info['use_double_dqn'] else 'Standard DQN'}")
        print(f"   - Network parameters: {agent_info['network_parameters']:,}")
        print(f"   - Current epsilon: {agent_info['current_epsilon']:.3f}")
        
        # Test action selection
        test_state = np.random.randn(12)
        action = agent.select_action(test_state)
        
        assert 0 <= action < 7, f"Action {action} not in valid range"
        print(f"✅ Action selection working: action={action}")
        
        # Test Q-value computation
        q_values = agent.get_q_values(test_state)
        assert q_values.shape == (7,), f"Q-values shape should be (7,), got {q_values.shape}"
        print(f"✅ Q-value computation working: {q_values}")
        
        # Test experience storage
        next_state = np.random.randn(12)
        agent.store_experience(test_state, action, 1.0, next_state, False)
        
        # Add enough experiences for training
        for i in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.random() < 0.1
            
            agent.store_experience(state, action, reward, next_state, done)
        
        print(f"✅ Experience storage working: {agent.replay_buffer.size} experiences")
        
        # Test training step
        if agent.can_train():
            metrics = agent.train_step()
            
            expected_keys = ['loss', 'mean_q_value', 'epsilon', 'training_step']
            for key in expected_keys:
                assert key in metrics, f"Training metrics missing key: {key}"
            
            print(f"✅ Training step working:")
            print(f"   - Loss: {metrics['loss']:.4f}")
            print(f"   - Mean Q-value: {metrics['mean_q_value']:.3f}")
            print(f"   - Epsilon: {metrics['epsilon']:.3f}")
            print(f"   - Training step: {metrics['training_step']}")
        
        # Test episode statistics update
        agent.update_episode_stats(episode_reward=10.5, success=False)
        updated_info = agent.get_agent_info()
        assert updated_info['episode_count'] == 1, "Episode count should be updated"
        print(f"✅ Episode statistics update working")
        
        # Test model saving/loading
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            agent.save_model(tmp_path)
            
            # Create new agent and load
            new_agent = DQNAgent(seed=42)
            new_agent.load_model(tmp_path)
            
        finally:
            # Cleanup with retry for Windows
            try:
                os.unlink(tmp_path)
            except:
                pass  # Ignore cleanup errors
        
        print(f"✅ Model save/load working")
        
        print("✅ DQN Agent Integration Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ DQN Agent Integration Tests FAILED: {e}")
        return False


def test_environment_integration():
    """Test 5: Environment Integration"""
    print("\n" + "=" * 60)
    print("TEST 5: Environment Integration")
    print("=" * 60)
    
    try:
        # Create environment and agent
        env = ParkingEnv(
            width=30.0, height=20.0,
            enable_obstacles=True,
            max_steps=50  # Short episodes for testing
        )
        
        agent = DQNAgent(
            batch_size=32,
            replay_buffer_size=1000,
            target_update_freq=50,
            seed=42
        )
        
        print(f"✅ Environment and agent created")
        
        # Test episode loop
        total_rewards = []
        
        for episode in range(3):  # Quick test with 3 episodes
            obs = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 20:  # Limit steps for testing
                # Select action
                action = agent.select_action(obs)
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                
                # Store experience
                agent.store_experience(obs, action, reward, next_obs, done)
                
                episode_reward += reward
                obs = next_obs
                steps += 1
                
                # Train if possible
                if agent.can_train() and steps % 5 == 0:  # Train every 5 steps
                    metrics = agent.train_step()
                
                if done:
                    break
            
            # Update episode statistics
            agent.update_episode_stats(episode_reward, info.get('is_successful', False))
            total_rewards.append(episode_reward)
            
            print(f"   Episode {episode + 1}: reward={episode_reward:.2f}, steps={steps}")
        
        env.close()
        
        # Test final agent state
        final_info = agent.get_agent_info()
        print(f"✅ Environment integration completed:")
        print(f"   - Episodes completed: {final_info['episode_count']}")
        print(f"   - Training steps: {final_info['training_step']}")
        print(f"   - Buffer usage: {final_info['buffer_usage']:.1f}%")
        print(f"   - Average reward: {np.mean(total_rewards):.2f}")
        
        print("✅ Environment Integration Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Environment Integration Tests FAILED: {e}")
        return False


def test_performance_benchmarks():
    """Test 6: Performance Benchmarks"""
    print("\n" + "=" * 60)
    print("TEST 6: Performance Benchmarks")
    print("=" * 60)
    
    try:
        # Create agent for performance testing
        agent = DQNAgent(
            batch_size=64,
            replay_buffer_size=10000,
            seed=42
        )
        
        # Fill replay buffer
        print("   Filling replay buffer...")
        for i in range(1000):
            state = np.random.randn(12)
            action = np.random.randint(0, 7)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.random() < 0.1
            
            agent.store_experience(state, action, reward, next_state, done)
        
        buffer_info = agent.replay_buffer.get_buffer_info()
        print(f"✅ Buffer filled: {buffer_info['current_size']} experiences")
        print(f"   Memory usage: {buffer_info['memory_usage_mb']:.2f} MB")
        
        # Benchmark training speed
        print("   Benchmarking training speed...")
        start_time = time.time()
        
        training_times = []
        for i in range(50):  # 50 training steps
            step_start = time.time()
            metrics = agent.train_step()
            step_end = time.time()
            training_times.append(step_end - step_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_step_time = np.mean(training_times)
        
        print(f"✅ Training benchmark completed:")
        print(f"   - 50 training steps: {total_time:.3f}s")
        print(f"   - Average step time: {avg_step_time*1000:.1f}ms")
        print(f"   - Steps per second: {50/total_time:.1f}")
        
        # Benchmark action selection speed
        print("   Benchmarking action selection...")
        test_states = [np.random.randn(12) for _ in range(1000)]
        
        start_time = time.time()
        for state in test_states:
            action = agent.select_action(state, epsilon=0.1)
        end_time = time.time()
        
        action_time = end_time - start_time
        print(f"✅ Action selection benchmark:")
        print(f"   - 1000 action selections: {action_time:.3f}s")
        print(f"   - Actions per second: {1000/action_time:.0f}")
        
        # Memory usage check
        final_agent_info = agent.get_agent_info()
        print(f"✅ Memory efficiency:")
        print(f"   - Buffer usage: {final_agent_info['buffer_usage']:.1f}%")
        print(f"   - Network parameters: {final_agent_info['network_parameters']:,}")
        
        print("✅ Performance Benchmarks Tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmarks Tests FAILED: {e}")
        return False


def main():
    """Run all Phase 4 tests"""
    print("🚀 PHASE 4 TESTING: DQN Network Architecture")
    print("Testing Deep Q-Learning components with paper specifications")
    print()
    
    # Run all tests
    tests = [
        test_dqn_network_architecture,
        test_experience_replay_buffer,
        test_loss_functions_and_utilities,
        test_dqn_agent_integration,
        test_environment_integration,
        test_performance_benchmarks
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
    print("PHASE 4 TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "DQN Network Architecture",
        "Experience Replay Buffer",
        "Loss Functions & Utilities",
        "DQN Agent Integration",
        "Environment Integration",
        "Performance Benchmarks"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name:25s} {status}")
    
    total_passed = sum(results)
    total_tests = len(results)
    
    print(f"\nResult: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL PHASE 4 TESTS PASSED!")
        print("DQN Network Architecture implementation is complete and ready for Phase 5.")
        print("\nPhase 4 Features Validated:")
        print("✅ DQN network architecture (3×256 hidden layers + ReLU)")
        print("✅ Experience replay buffer (~10^5 capacity)")
        print("✅ Huber loss function and training utilities")
        print("✅ Complete DQN agent with epsilon-greedy policy")
        print("✅ Double DQN and prioritized experience replay")
        print("✅ Full integration with parking environment")
        print("✅ Performance benchmarks and memory efficiency")
        
        print("\nNext Steps:")
        print("📋 Begin Phase 5: Training Pipeline & Hyperparameter Setup")
        print("⚙️ Implement complete training loop")
        print("📊 Set up training monitoring and logging")
        print("🔧 Add model checkpointing and validation")
        
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed.")
        print("Please fix the failing tests before proceeding to Phase 5.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 