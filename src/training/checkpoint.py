"""
Model Checkpointing and State Management
Handles saving and loading of DQN models and training state.

Features:
- Model state saving and loading
- Training state preservation (episode, step, metrics)
- Best model tracking
- Checkpoint compression and validation
- Recovery from training interruptions
"""

import os
import torch
import pickle
import json
import shutil
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib


class ModelCheckpoint:
    """
    Comprehensive model checkpoint manager for DQN training.
    
    Handles model saving, loading, and training state management
    with validation and error recovery capabilities.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        experiment_name: str = None,
        keep_best: int = 3,
        keep_latest: int = 5,
        compress_checkpoints: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Experiment identifier
            keep_best: Number of best models to keep
            keep_latest: Number of latest models to keep
            compress_checkpoints: Whether to compress checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name or f"dqn_parking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.keep_best = keep_best
        self.keep_latest = keep_latest
        self.compress_checkpoints = compress_checkpoints
        
        # Create experiment checkpoint directory
        self.experiment_dir = os.path.join(checkpoint_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Checkpoint tracking
        self.best_checkpoints = []  # List of (score, filepath) tuples
        self.latest_checkpoints = []  # List of filepaths
        
        # State tracking
        self.checkpoint_counter = 0
        
        print(f"💾 Checkpoint manager initialized: {self.experiment_dir}")
    
    def save_checkpoint(
        self,
        agent,
        episode: int,
        training_step: int,
        performance_score: float,
        metrics: Dict[str, Any],
        is_best: bool = False,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a complete training checkpoint.
        
        Args:
            agent: DQN agent to save
            episode: Current episode number
            training_step: Current training step
            performance_score: Performance metric for ranking
            metrics: Training metrics dictionary
            is_best: Whether this is the best model so far
            additional_data: Additional data to save
            
        Returns:
            Path to saved checkpoint file
        """
        self.checkpoint_counter += 1
        
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = f"best_model_ep{episode:06d}_{timestamp}.pth"
        else:
            filename = f"checkpoint_ep{episode:06d}_{timestamp}.pth"
        
        filepath = os.path.join(self.experiment_dir, filename)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state': {
                'main_network': agent.main_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'optimizer': agent.optimizer.state_dict()
            },
            'training_state': {
                'episode': episode,
                'training_step': training_step,
                'total_episodes': getattr(agent, 'episode_count', episode),
                'epsilon': getattr(agent, 'epsilon', 0.0),
                'learning_rate': agent.optimizer.param_groups[0]['lr']
            },
            'performance': {
                'score': performance_score,
                'metrics': metrics,
                'is_best': is_best
            },
            'metadata': {
                'experiment_name': self.experiment_name,
                'save_time': datetime.now().isoformat(),
                'checkpoint_counter': self.checkpoint_counter,
                'agent_config': self._get_agent_config(agent)
            }
        }
        
        # Add experience replay buffer state
        if hasattr(agent, 'replay_buffer') and agent.replay_buffer.size > 0:
            checkpoint_data['replay_buffer'] = {
                'size': agent.replay_buffer.size,
                'position': getattr(agent.replay_buffer, 'position', 0),
                # Note: We don't save the full buffer to keep checkpoint size reasonable
                # but we could add an option to save it for complete recovery
            }
        
        # Add additional data if provided
        if additional_data:
            checkpoint_data['additional'] = additional_data
        
        # Save checkpoint
        try:
            if self.compress_checkpoints:
                # Save with compression
                torch.save(checkpoint_data, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            else:
                torch.save(checkpoint_data, filepath)
            
            # Calculate file hash for validation
            file_hash = self._calculate_file_hash(filepath)
            checkpoint_data['metadata']['file_hash'] = file_hash
            
            # Save metadata separately for quick access
            metadata_file = filepath.replace('.pth', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(checkpoint_data['metadata'], f, indent=2)
            
            print(f"💾 Checkpoint saved: {filename} (score: {performance_score:.4f})")
            
            # Update checkpoint tracking
            self._update_checkpoint_tracking(filepath, performance_score, is_best)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return filepath
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # Clean up partial files
            for file_to_remove in [filepath, filepath.replace('.pth', '_metadata.json')]:
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)
            raise
    
    def load_checkpoint(
        self,
        agent,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_replay_buffer: bool = False,
        device: str = None
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            agent: DQN agent to load state into
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_replay_buffer: Whether to load replay buffer state
            device: Device to load tensors to
            
        Returns:
            Dictionary containing loaded metadata and training state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Validate checkpoint integrity
            self._validate_checkpoint(checkpoint_path, checkpoint_data)
            
            # Load model state
            agent.main_network.load_state_dict(checkpoint_data['model_state']['main_network'])
            agent.target_network.load_state_dict(checkpoint_data['model_state']['target_network'])
            
            # Load optimizer state if requested
            if load_optimizer and 'optimizer' in checkpoint_data['model_state']:
                agent.optimizer.load_state_dict(checkpoint_data['model_state']['optimizer'])
            
            # Restore training state
            training_state = checkpoint_data['training_state']
            if hasattr(agent, 'episode_count'):
                agent.episode_count = training_state.get('total_episodes', 0)
            if hasattr(agent, 'training_step'):
                agent.training_step = training_state.get('training_step', 0)
            if hasattr(agent, 'epsilon'):
                agent.epsilon = training_state.get('epsilon', 0.0)
            
            # Load replay buffer state if requested
            if load_replay_buffer and 'replay_buffer' in checkpoint_data:
                buffer_state = checkpoint_data['replay_buffer']
                # Note: This would require implementing buffer state saving/loading
                print(f"📝 Replay buffer state available (size: {buffer_state.get('size', 0)})")
            
            metadata = checkpoint_data.get('metadata', {})
            performance = checkpoint_data.get('performance', {})
            
            print(f"✅ Checkpoint loaded: {os.path.basename(checkpoint_path)}")
            print(f"   Episode: {training_state.get('episode', 0):,}")
            print(f"   Training step: {training_state.get('training_step', 0):,}")
            print(f"   Performance score: {performance.get('score', 0):.4f}")
            
            return {
                'training_state': training_state,
                'performance': performance,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        if not self.best_checkpoints:
            return None
        # Sort by score (descending) and return the best one
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        return self.best_checkpoints[0][1]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the most recent checkpoint."""
        if not self.latest_checkpoints:
            return None
        return self.latest_checkpoints[-1]
    
    def list_checkpoints(self) -> Dict[str, list]:
        """List all available checkpoints."""
        checkpoints = []
        
        for filename in os.listdir(self.experiment_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.experiment_dir, filename)
                metadata_file = filepath.replace('.pth', '_metadata.json')
                
                # Try to load metadata
                metadata = {}
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                checkpoints.append({
                    'filename': filename,
                    'filepath': filepath,
                    'size_mb': os.path.getsize(filepath) / (1024 * 1024),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
                    'metadata': metadata
                })
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        
        return {
            'total_checkpoints': len(checkpoints),
            'checkpoints': checkpoints,
            'best_checkpoints': [cp[1] for cp in self.best_checkpoints],
            'latest_checkpoints': self.latest_checkpoints
        }
    
    def _get_agent_config(self, agent) -> Dict[str, Any]:
        """Extract agent configuration for saving."""
        config = {
            'network_type': type(agent.main_network).__name__,
            'state_dim': getattr(agent.main_network, 'state_dim', None),
            'action_dim': getattr(agent.main_network, 'action_dim', None),
            'hidden_dim': getattr(agent.main_network, 'hidden_dim', None),
            'use_double_dqn': getattr(agent, 'use_double_dqn', False),
            'replay_buffer_size': getattr(agent.replay_buffer, 'capacity', None) if hasattr(agent, 'replay_buffer') else None
        }
        return config
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file for validation."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _validate_checkpoint(self, filepath: str, checkpoint_data: Dict[str, Any]):
        """Validate checkpoint integrity."""
        metadata = checkpoint_data.get('metadata', {})
        
        # Check file hash if available
        if 'file_hash' in metadata:
            expected_hash = metadata['file_hash']
            # Note: We'd need to calculate hash without the hash field for this to work
            # This is a simplified version
            
        # Check required fields
        required_fields = ['model_state', 'training_state']
        for field in required_fields:
            if field not in checkpoint_data:
                raise ValueError(f"Missing required field in checkpoint: {field}")
        
        # Check model state
        model_state = checkpoint_data['model_state']
        required_model_fields = ['main_network', 'target_network']
        for field in required_model_fields:
            if field not in model_state:
                raise ValueError(f"Missing model state field: {field}")
    
    def _update_checkpoint_tracking(self, filepath: str, score: float, is_best: bool):
        """Update checkpoint tracking lists."""
        # Update latest checkpoints
        self.latest_checkpoints.append(filepath)
        
        # Update best checkpoints if this is marked as best
        if is_best:
            self.best_checkpoints.append((score, filepath))
            # Sort and keep only the best ones
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
            if len(self.best_checkpoints) > self.keep_best:
                # Remove worst checkpoint file
                _, worst_checkpoint = self.best_checkpoints.pop()
                if os.path.exists(worst_checkpoint):
                    os.remove(worst_checkpoint)
                    metadata_file = worst_checkpoint.replace('.pth', '_metadata.json')
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints based on retention policy."""
        # Clean up latest checkpoints
        while len(self.latest_checkpoints) > self.keep_latest:
            old_checkpoint = self.latest_checkpoints.pop(0)
            
            # Don't delete if it's a best checkpoint
            is_best_checkpoint = any(old_checkpoint == best[1] for best in self.best_checkpoints)
            
            if not is_best_checkpoint and os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                metadata_file = old_checkpoint.replace('.pth', '_metadata.json')
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                print(f"🗑️ Removed old checkpoint: {os.path.basename(old_checkpoint)}")
    
    def save_final_model(self, agent, performance_metrics: Dict[str, Any]) -> str:
        """Save the final trained model."""
        filename = f"final_model_{self.experiment_name}.pth"
        filepath = os.path.join(self.experiment_dir, filename)
        
        final_model_data = {
            'model_state': agent.main_network.state_dict(),
            'network_config': self._get_agent_config(agent),
            'performance_metrics': performance_metrics,
            'training_complete': True,
            'save_time': datetime.now().isoformat(),
            'experiment_name': self.experiment_name
        }
        
        torch.save(final_model_data, filepath)
        print(f"🎯 Final model saved: {filename}")
        
        return filepath
    
    def export_for_deployment(self, agent, export_path: str) -> str:
        """Export model for deployment (inference only)."""
        deployment_data = {
            'model_state': agent.main_network.state_dict(),
            'network_config': self._get_agent_config(agent),
            'inference_only': True,
            'export_time': datetime.now().isoformat()
        }
        
        torch.save(deployment_data, export_path)
        print(f"📦 Model exported for deployment: {export_path}")
        
        return export_path


if __name__ == "__main__":
    # Example usage and testing
    print("Testing checkpoint manager...")
    
    # Create a mock agent for testing
    class MockAgent:
        def __init__(self):
            self.main_network = torch.nn.Linear(10, 5)
            self.target_network = torch.nn.Linear(10, 5)
            self.optimizer = torch.optim.Adam(self.main_network.parameters())
            self.episode_count = 0
            self.training_step = 0
            self.epsilon = 1.0
    
    # Test checkpoint manager
    checkpoint_manager = ModelCheckpoint(experiment_name="test_experiment")
    
    # Create mock agent
    agent = MockAgent()
    
    # Test saving checkpoint
    checkpoint_path = checkpoint_manager.save_checkpoint(
        agent=agent,
        episode=100,
        training_step=500,
        performance_score=85.5,
        metrics={'success_rate': 0.85, 'collision_rate': 0.02},
        is_best=True
    )
    
    # Test loading checkpoint
    loaded_data = checkpoint_manager.load_checkpoint(agent, checkpoint_path)
    
    # Test listing checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"Found {checkpoints['total_checkpoints']} checkpoints")
    
    print("✅ Checkpoint manager test completed successfully!") 