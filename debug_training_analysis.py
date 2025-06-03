"""
Training Analysis Script
Analyzes the training logs to identify issues and understand agent behavior.
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def load_training_logs(experiment_dir: str) -> Dict[str, Any]:
    """Load all training logs from an experiment directory."""
    
    logs = {
        'episodes': [],
        'training_steps': [],
        'evaluations': []
    }
    
    # Load episode logs from CSV
    episode_csv_file = os.path.join(experiment_dir, 'episodes.csv')
    if os.path.exists(episode_csv_file):
        df = pd.read_csv(episode_csv_file)
        logs['episodes'] = df.to_dict('records')
    
    # Load training step logs from CSV
    training_csv_file = os.path.join(experiment_dir, 'training.csv')
    if os.path.exists(training_csv_file):
        df = pd.read_csv(training_csv_file)
        logs['training_steps'] = df.to_dict('records')
    
    # Load evaluation logs from JSON (if exists)
    eval_log_file = os.path.join(experiment_dir, 'evaluation_logs.json')
    if os.path.exists(eval_log_file):
        with open(eval_log_file, 'r') as f:
            for line in f:
                if line.strip():
                    logs['evaluations'].append(json.loads(line))
    
    return logs

def analyze_episode_progression(episodes: List[Dict]) -> Dict[str, Any]:
    """Analyze episode-by-episode progression."""
    
    if not episodes:
        return {'error': 'No episode data found'}
    
    # Extract metrics
    rewards = [ep['total_reward'] for ep in episodes]
    steps = [ep['steps'] for ep in episodes]
    successes = [ep['success'] for ep in episodes]
    collisions = [ep['collision'] for ep in episodes]
    distances = [ep.get('distance_to_target', float('inf')) for ep in episodes]
    exploration_rates = [ep.get('exploration_rate', 0) for ep in episodes]
    
    # Calculate moving averages
    window = min(10, len(rewards) // 2)
    if window > 0:
        reward_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        success_rate_ma = np.convolve(successes, np.ones(window)/window, mode='valid')
    else:
        reward_ma = rewards
        success_rate_ma = successes
    
    analysis = {
        'total_episodes': len(episodes),
        'final_success_rate': np.mean(successes[-10:]) if len(successes) >= 10 else np.mean(successes),
        'final_collision_rate': np.mean(collisions[-10:]) if len(collisions) >= 10 else np.mean(collisions),
        'reward_trend': {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'final_avg': np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
            'improvement': (np.mean(rewards[-10:]) - np.mean(rewards[:10])) if len(rewards) >= 20 else 0
        },
        'steps_analysis': {
            'mean': np.mean(steps),
            'std': np.std(steps),
            'min': np.min(steps),
            'max': np.max(steps)
        },
        'distance_analysis': {
            'mean': np.mean([d for d in distances if d != float('inf')]),
            'min': np.min([d for d in distances if d != float('inf')]) if any(d != float('inf') for d in distances) else float('inf'),
            'final_avg': np.mean([d for d in distances[-10:] if d != float('inf')]) if len(distances) >= 10 else np.mean([d for d in distances if d != float('inf')])
        },
        'exploration_analysis': {
            'start_epsilon': exploration_rates[0] if exploration_rates else 0,
            'end_epsilon': exploration_rates[-1] if exploration_rates else 0,
            'decay_rate': (exploration_rates[0] - exploration_rates[-1]) / len(exploration_rates) if len(exploration_rates) > 1 else 0
        }
    }
    
    return analysis

def analyze_training_steps(training_steps: List[Dict]) -> Dict[str, Any]:
    """Analyze training step progression."""
    
    if not training_steps:
        return {'error': 'No training step data found'}
    
    # Extract metrics
    losses = [step['loss'] for step in training_steps]
    q_values = [step.get('mean_q_value', 0) for step in training_steps]
    grad_norms = [step.get('gradient_norm', 0) for step in training_steps]
    learning_rates = [step.get('learning_rate', 0) for step in training_steps]
    
    analysis = {
        'total_training_steps': len(training_steps),
        'loss_analysis': {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'min': np.min(losses),
            'max': np.max(losses),
            'final_avg': np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses),
            'convergence': np.std(losses[-100:]) if len(losses) >= 100 else np.std(losses)
        },
        'q_value_analysis': {
            'mean': np.mean(q_values),
            'std': np.std(q_values),
            'min': np.min(q_values),
            'max': np.max(q_values),
            'final_avg': np.mean(q_values[-100:]) if len(q_values) >= 100 else np.mean(q_values)
        },
        'gradient_analysis': {
            'mean': np.mean(grad_norms),
            'std': np.std(grad_norms),
            'max': np.max(grad_norms)
        }
    }
    
    return analysis

def identify_issues(episode_analysis: Dict, training_analysis: Dict) -> List[str]:
    """Identify potential issues with training."""
    
    issues = []
    
    # Check success rate
    if episode_analysis.get('final_success_rate', 0) < 0.1:
        issues.append("❌ Very low success rate (<10%)")
    
    # Check collision rate
    if episode_analysis.get('final_collision_rate', 0) > 0.5:
        issues.append("❌ High collision rate (>50%)")
    
    # Check reward improvement
    if episode_analysis.get('reward_trend', {}).get('improvement', 0) < 0:
        issues.append("❌ Rewards are decreasing over time")
    
    # Check loss convergence
    if training_analysis.get('loss_analysis', {}).get('convergence', float('inf')) > 1.0:
        issues.append("⚠️ Loss is not converging (high variance)")
    
    # Check Q-values
    q_mean = training_analysis.get('q_value_analysis', {}).get('mean', 0)
    if abs(q_mean) > 100:
        issues.append("⚠️ Q-values are unstable (very large magnitude)")
    
    # Check gradient norms
    grad_max = training_analysis.get('gradient_analysis', {}).get('max', 0)
    if grad_max > 10:
        issues.append("⚠️ Large gradient norms detected (possible gradient explosion)")
    
    # Check distance to target
    dist_mean = episode_analysis.get('distance_analysis', {}).get('mean', float('inf'))
    if dist_mean == float('inf') or dist_mean > 30:
        issues.append("❌ Agent not getting close to target")
    
    return issues

def create_diagnostic_plots(logs: Dict[str, Any], save_dir: str):
    """Create diagnostic plots for training analysis."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    episodes = logs['episodes']
    training_steps = logs['training_steps']
    
    if episodes:
        # Episode metrics plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Episode Progression Analysis', fontsize=16)
        
        # Rewards
        rewards = [ep['total_reward'] for ep in episodes]
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Success/Collision rates
        successes = [ep['success'] for ep in episodes]
        collisions = [ep['collision'] for ep in episodes]
        window = 10
        if len(successes) >= window:
            success_ma = np.convolve(successes, np.ones(window)/window, mode='valid')
            collision_ma = np.convolve(collisions, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(successes)), success_ma, label='Success Rate')
            axes[0, 1].plot(range(window-1, len(collisions)), collision_ma, label='Collision Rate')
        else:
            axes[0, 1].plot(successes, label='Success Rate')
            axes[0, 1].plot(collisions, label='Collision Rate')
        axes[0, 1].set_title('Success/Collision Rates (Moving Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].legend()
        
        # Distance to target
        distances = [ep.get('distance_to_target', float('inf')) for ep in episodes]
        valid_distances = [d if d != float('inf') else np.nan for d in distances]
        axes[0, 2].plot(valid_distances)
        axes[0, 2].set_title('Distance to Target')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Distance (m)')
        
        # Episode steps
        steps = [ep['steps'] for ep in episodes]
        axes[1, 0].plot(steps)
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Exploration rate
        exploration = [ep.get('exploration_rate', 0) for ep in episodes]
        axes[1, 1].plot(exploration)
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        # Parking accuracy
        parking_acc = [ep.get('parking_accuracy', 0) for ep in episodes]
        axes[1, 2].plot(parking_acc)
        axes[1, 2].set_title('Parking Accuracy')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'episode_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    if training_steps:
        # Training metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Step Analysis', fontsize=16)
        
        # Loss
        losses = [step['loss'] for step in training_steps]
        axes[0, 0].plot(losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_yscale('log')
        
        # Q-values
        q_values = [step.get('mean_q_value', 0) for step in training_steps]
        axes[0, 1].plot(q_values)
        axes[0, 1].set_title('Mean Q-Values')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Q-Value')
        
        # Gradient norms
        grad_norms = [step.get('gradient_norm', 0) for step in training_steps]
        axes[1, 0].plot(grad_norms)
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        
        # Learning rate
        lrs = [step.get('learning_rate', 0) for step in training_steps]
        axes[1, 1].plot(lrs)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function."""
    
    # Find the most recent experiment
    log_dir = "data/training_logs"
    if not os.path.exists(log_dir):
        print("❌ No training logs found. Run training first.")
        return
    
    # Get all experiment directories
    experiments = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    
    if not experiments:
        print("❌ No experiment directories found.")
        return
    
    print(f"📁 Found experiments: {experiments}")
    
    # Use the most recent experiment (or force specific one)
    experiments.sort()
    latest_experiment = "dqn_parking_1748940537"  # Force the visualization training
    if latest_experiment not in experiments:
        latest_experiment = experiments[-1]
    experiment_dir = os.path.join(log_dir, latest_experiment)
    
    print(f"🔍 Analyzing experiment: {latest_experiment}")
    print(f"📁 Directory: {experiment_dir}")
    
    # Load logs
    logs = load_training_logs(experiment_dir)
    
    print(f"\n📊 Data Summary:")
    print(f"   Episodes: {len(logs['episodes'])}")
    print(f"   Training Steps: {len(logs['training_steps'])}")
    print(f"   Evaluations: {len(logs['evaluations'])}")
    
    # Analyze episodes
    if logs['episodes']:
        episode_analysis = analyze_episode_progression(logs['episodes'])
        print(f"\n📈 Episode Analysis:")
        print(f"   Total Episodes: {episode_analysis['total_episodes']}")
        print(f"   Final Success Rate: {episode_analysis['final_success_rate']:.1%}")
        print(f"   Final Collision Rate: {episode_analysis['final_collision_rate']:.1%}")
        print(f"   Reward Trend: {episode_analysis['reward_trend']['final_avg']:.2f} (avg)")
        print(f"   Reward Improvement: {episode_analysis['reward_trend']['improvement']:.2f}")
        print(f"   Final Distance to Target: {episode_analysis['distance_analysis']['final_avg']:.2f}m")
    else:
        episode_analysis = {}
    
    # Analyze training steps
    if logs['training_steps']:
        training_analysis = analyze_training_steps(logs['training_steps'])
        print(f"\n🧠 Training Analysis:")
        print(f"   Total Training Steps: {training_analysis['total_training_steps']}")
        print(f"   Final Loss: {training_analysis['loss_analysis']['final_avg']:.4f}")
        print(f"   Loss Convergence (std): {training_analysis['loss_analysis']['convergence']:.4f}")
        print(f"   Mean Q-Values: {training_analysis['q_value_analysis']['final_avg']:.2f}")
        print(f"   Max Gradient Norm: {training_analysis['gradient_analysis']['max']:.2f}")
    else:
        training_analysis = {}
    
    # Identify issues
    issues = identify_issues(episode_analysis, training_analysis)
    
    if issues:
        print(f"\n⚠️ Identified Issues:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✅ No major issues identified!")
    
    # Create diagnostic plots
    print(f"\n📊 Creating diagnostic plots...")
    create_diagnostic_plots(logs, os.path.join(experiment_dir, "analysis"))
    print(f"   Plots saved to: {os.path.join(experiment_dir, 'analysis')}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if episode_analysis.get('final_success_rate', 0) < 0.1:
        print("   🎯 Increase training episodes or adjust reward function")
        print("   🔧 Consider reducing environment complexity initially")
    
    if episode_analysis.get('final_collision_rate', 0) > 0.3:
        print("   🛡️ Add stronger collision penalties")
        print("   📏 Check sensor readings and obstacle detection")
    
    if training_analysis.get('loss_analysis', {}).get('convergence', 0) > 0.5:
        print("   📉 Reduce learning rate for more stable training")
        print("   🎯 Consider increasing batch size")
    
    if episode_analysis.get('distance_analysis', {}).get('final_avg', float('inf')) > 20:
        print("   🗺️ Agent may not be learning navigation properly")
        print("   🔧 Check reward shaping for distance-based progress")


if __name__ == "__main__":
    main() 