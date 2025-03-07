"""
ML Integration Module
===================

Provides integration between classical machine learning models and quantum circuits.
"""

from typing import List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from .quantum_circuit import QuantumLayer

class HybridModel(nn.Module):
    """A hybrid quantum-classical neural network model."""
    
    def __init__(
        self,
        input_size: int,
        n_qubits: int,
        quantum_depth: int = 1,
        classical_layers: Optional[List[int]] = None
    ):
        """
        Initialize a hybrid quantum-classical model.
        
        Args:
            input_size: Dimension of input features
            n_qubits: Number of qubits in quantum circuit
            quantum_depth: Depth of quantum circuit (default: 1)
            classical_layers: List of classical layer sizes (default: None)
        """
        super().__init__()
        self.input_size = input_size
        self.n_qubits = n_qubits
        self.quantum_depth = quantum_depth
        
        # Classical pre-processing layers
        layers = []
        current_size = input_size
        
        if classical_layers:
            for layer_size in classical_layers:
                layers.extend([
                    nn.Linear(current_size, layer_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_size)
                ])
                current_size = layer_size
        
        # Final classical layer to match quantum parameters
        n_quantum_params = n_qubits * quantum_depth * 3  # 3 parameters per qubit per layer
        layers.append(nn.Linear(current_size, n_quantum_params))
        
        self.classical_net = nn.Sequential(*layers)
        self.quantum_layer = QuantumLayer(n_qubits, quantum_depth)
        
        # Post-processing layer
        self.post_process = nn.Linear(n_qubits, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        # Classical pre-processing
        classical_out = self.classical_net(x)
        
        # Quantum processing
        quantum_params = classical_out.detach().numpy()
        quantum_out = []
        
        # Process each sample in the batch
        for params in quantum_params:
            result = self.quantum_layer.forward(params.tolist())
            quantum_out.append(result)
            
        # Convert quantum results to tensor
        quantum_tensor = torch.tensor(quantum_out, dtype=torch.float32)
        
        # Post-processing
        output = self.post_process(quantum_tensor)
        return output

class QuantumOptimizer:
    """Optimizer for quantum-classical hybrid models."""
    
    def __init__(
        self,
        model: HybridModel,
        loss_fn: Callable,
        classical_optimizer: torch.optim.Optimizer,
        **kwargs
    ):
        """
        Initialize the quantum optimizer.
        
        Args:
            model: Hybrid quantum-classical model
            loss_fn: Loss function
            classical_optimizer: Classical optimizer for the model
            **kwargs: Additional optimizer parameters
        """
        self.model = model
        self.loss_fn = loss_fn
        self.classical_optimizer = classical_optimizer
        
    def step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one optimization step.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Tuple of (predictions, loss)
        """
        # Forward pass
        self.classical_optimizer.zero_grad()
        y_pred = self.model(x_batch)
        
        # Compute loss
        loss = self.loss_fn(y_pred, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Update classical parameters
        self.classical_optimizer.step()
        
        return y_pred, loss
