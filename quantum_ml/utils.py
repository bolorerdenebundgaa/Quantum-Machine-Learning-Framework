"""
Utilities Module
==============

Helper functions for data preprocessing, visualization, and quantum state analysis.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
import torch

def encode_classical_data(
    data: Union[np.ndarray, torch.Tensor],
    encoding: str = 'angle'
) -> np.ndarray:
    """
    Encode classical data for quantum processing.
    
    Args:
        data: Input data array
        encoding: Encoding method ('angle', 'amplitude', or 'phase')
        
    Returns:
        Encoded data as numpy array
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    if encoding == 'angle':
        # Map data to rotation angles [0, 2π]
        return 2 * np.pi * (data - np.min(data)) / (np.max(data) - np.min(data))
    elif encoding == 'amplitude':
        # Normalize for amplitude encoding
        return data / np.sqrt(np.sum(np.abs(data) ** 2))
    elif encoding == 'phase':
        # Map to phase [0, 2π]
        return np.exp(1j * data)
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")

def visualize_quantum_state(
    circuit: QuantumCircuit,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Visualize the quantum state using various representations.
    
    Args:
        circuit: Quantum circuit to visualize
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    
    # Draw circuit
    plt.subplot(1, 2, 1)
    circuit.draw(output='mpl')
    plt.title('Quantum Circuit')
    
    # TODO: Add Bloch sphere visualization when statevector is available
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, 'Bloch Sphere\n(Coming Soon)', 
             ha='center', va='center')
    plt.title('State Visualization')
    
    plt.tight_layout()
    plt.show()

def create_quantum_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    encoding: str = 'angle'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a quantum-ready dataset from classical data.
    
    Args:
        X: Input features
        y: Target values
        n_qubits: Number of qubits to encode into
        encoding: Data encoding method
        
    Returns:
        Tuple of (encoded_features, targets) as PyTorch tensors
    """
    if X.shape[1] > 2**n_qubits:
        raise ValueError(
            f"Input dimension {X.shape[1]} too large for {n_qubits} qubits"
        )
    
    # Pad or truncate features to match qubit dimension
    target_dim = 2**n_qubits
    if X.shape[1] < target_dim:
        pad_width = ((0, 0), (0, target_dim - X.shape[1]))
        X = np.pad(X, pad_width, mode='constant')
    
    # Encode data
    X_encoded = encode_classical_data(X, encoding)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_encoded)
    y_tensor = torch.FloatTensor(y)
    
    return X_tensor, y_tensor

def calculate_quantum_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> dict:
    """
    Calculate metrics specific to quantum ML models.
    
    Args:
        predictions: Model predictions
        targets: True target values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Classical metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    metrics['mse'] = mse
    metrics['mae'] = mae
    
    # TODO: Add quantum-specific metrics
    # e.g., fidelity, entanglement measures
    
    return metrics

def save_quantum_model(
    model: 'HybridModel',
    path: str,
    metadata: Optional[dict] = None
) -> None:
    """
    Save a quantum-classical hybrid model.
    
    Args:
        model: The hybrid model to save
        path: Save path
        metadata: Optional metadata to save with the model
    """
    save_dict = {
        'model_state': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'n_qubits': model.n_qubits,
            'quantum_depth': model.quantum_depth
        }
    }
    
    if metadata:
        save_dict['metadata'] = metadata
        
    torch.save(save_dict, path)

def load_quantum_model(path: str) -> 'HybridModel':
    """
    Load a saved quantum-classical hybrid model.
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded hybrid model
    """
    from .ml_integration import HybridModel
    
    save_dict = torch.load(path)
    config = save_dict['model_config']
    
    model = HybridModel(
        input_size=config['input_size'],
        n_qubits=config['n_qubits'],
        quantum_depth=config['quantum_depth']
    )
    
    model.load_state_dict(save_dict['model_state'])
    return model
