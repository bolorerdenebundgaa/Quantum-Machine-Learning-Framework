"""
Configuration Module
==================

Handles framework settings and quantum backend configurations.
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass
import json
import os
import torch

@dataclass
class QuantumBackendConfig:
    """Configuration for quantum backend."""
    name: str
    n_qubits: int
    simulator: bool = True
    noise_model: Optional[str] = None
    shots: int = 1000
    optimization_level: int = 1

@dataclass
class MLConfig:
    """Configuration for classical ML components."""
    batch_size: int = 32
    learning_rate: float = 0.001
    n_epochs: int = 100
    optimizer: str = 'adam'
    loss_function: str = 'mse'

class FrameworkConfig:
    """Main configuration class for the QuaNTUM ML framework."""
    
    def __init__(
        self,
        quantum_backend: Optional[Union[Dict, QuantumBackendConfig]] = None,
        ml_config: Optional[Union[Dict, MLConfig]] = None
    ):
        """
        Initialize framework configuration.
        
        Args:
            quantum_backend: Quantum backend configuration
            ml_config: Classical ML configuration
        """
        # Set quantum backend config
        if quantum_backend is None:
            self.quantum_backend = QuantumBackendConfig(
                name='qasm_simulator',
                n_qubits=4
            )
        elif isinstance(quantum_backend, dict):
            self.quantum_backend = QuantumBackendConfig(**quantum_backend)
        else:
            self.quantum_backend = quantum_backend
            
        # Set ML config
        if ml_config is None:
            self.ml_config = MLConfig()
        elif isinstance(ml_config, dict):
            self.ml_config = MLConfig(**ml_config)
        else:
            self.ml_config = ml_config
            
        # Framework-wide settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logging_level = 'INFO'
        self.save_dir = os.path.join(os.getcwd(), 'quantum_ml_outputs')
        
    def save(self, path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration file
        """
        config_dict = {
            'quantum_backend': {
                'name': self.quantum_backend.name,
                'n_qubits': self.quantum_backend.n_qubits,
                'simulator': self.quantum_backend.simulator,
                'noise_model': self.quantum_backend.noise_model,
                'shots': self.quantum_backend.shots,
                'optimization_level': self.quantum_backend.optimization_level
            },
            'ml_config': {
                'batch_size': self.ml_config.batch_size,
                'learning_rate': self.ml_config.learning_rate,
                'n_epochs': self.ml_config.n_epochs,
                'optimizer': self.ml_config.optimizer,
                'loss_function': self.ml_config.loss_function
            },
            'framework': {
                'device': self.device,
                'logging_level': self.logging_level,
                'save_dir': self.save_dir
            }
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'FrameworkConfig':
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Loaded configuration object
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        instance = cls(
            quantum_backend=config_dict['quantum_backend'],
            ml_config=config_dict['ml_config']
        )
        
        # Load framework settings
        instance.device = config_dict['framework']['device']
        instance.logging_level = config_dict['framework']['logging_level']
        instance.save_dir = config_dict['framework']['save_dir']
        
        return instance
    
    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Key-value pairs of parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.quantum_backend, key):
                setattr(self.quantum_backend, key, value)
            elif hasattr(self.ml_config, key):
                setattr(self.ml_config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

# Default configuration
default_config = FrameworkConfig()
