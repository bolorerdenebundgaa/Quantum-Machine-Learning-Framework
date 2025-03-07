# QuaNTUM ML Framework

A Python framework for integrating quantum computing with machine learning, providing tools for building and training hybrid quantum-classical models.

## Features

- Quantum circuit operations with parameterized gates
- Hybrid quantum-classical neural networks
- Multiple data encoding strategies for quantum processing
- Built-in visualization tools for quantum states
- Configurable quantum backend settings
- Integration with PyTorch for classical ML components
- Quantum-aware optimization and training utilities

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv quantum_ml_env
source quantum_ml_env/bin/activate  # On Unix/macOS
# or
.\quantum_ml_env\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install numpy torch qiskit scikit-learn matplotlib
```

## Usage

### Basic Example

```python
from quantum_ml.ml_integration import HybridModel
from quantum_ml.config import FrameworkConfig
from quantum_ml.utils import create_quantum_dataset

# Configure the framework
config = FrameworkConfig(
    quantum_backend={
        'name': 'qasm_simulator',
        'n_qubits': 4
    }
)

# Create a hybrid model
model = HybridModel(
    input_size=2,
    n_qubits=4,
    quantum_depth=2,
    classical_layers=[8, 4]
)

# Prepare quantum dataset
X_q, y_q = create_quantum_dataset(X, y, n_qubits=4)

# Train and evaluate
# See examples/binary_classification.py for complete training example
```

## Project Structure

```
quantum_ml/
├── __init__.py
├── quantum_circuit.py    # Quantum circuit operations
├── ml_integration.py     # Hybrid model implementations
├── utils.py             # Helper functions and utilities
└── config.py            # Framework configuration

examples/
└── binary_classification.py  # Example implementation
```

## Core Components

### QuantumLayer

The `QuantumLayer` class provides a quantum circuit that can be integrated with classical neural networks:

```python
from quantum_ml.quantum_circuit import QuantumLayer

# Create a quantum layer
quantum_layer = QuantumLayer(n_qubits=4, depth=2)

# Get the quantum circuit
circuit = quantum_layer.get_circuit()
```

### HybridModel

The `HybridModel` class combines classical and quantum processing:

```python
from quantum_ml.ml_integration import HybridModel

# Create a hybrid model
model = HybridModel(
    input_size=2,
    n_qubits=4,
    quantum_depth=2,
    classical_layers=[8, 4]
)
```

### Configuration

Use `FrameworkConfig` to manage framework settings:

```python
from quantum_ml.config import FrameworkConfig

config = FrameworkConfig(
    quantum_backend={
        'name': 'qasm_simulator',
        'n_qubits': 4,
        'shots': 1000
    },
    ml_config={
        'batch_size': 32,
        'learning_rate': 0.01
    }
)
```

## Examples

See the `examples/` directory for complete implementation examples:

- `binary_classification.py`: Demonstrates binary classification using a hybrid quantum-classical model

## Use Cases and Examples

### 1. Binary Classification
Perfect for tasks like medical diagnosis or fraud detection where quantum advantage might be possible.

```python
from quantum_ml.ml_integration import HybridModel
from quantum_ml.utils import create_quantum_dataset
import numpy as np

# Example: Medical diagnosis
def medical_diagnosis_example():
    # Simulated medical data
    n_samples = 1000
    n_features = 10
    X_medical = np.random.randn(n_samples, n_features)  # Medical features
    y_diagnosis = (X_medical.sum(axis=1) > 0).astype(int)  # Binary diagnosis
    
    # Create quantum dataset
    X_q, y_q = create_quantum_dataset(
        X_medical, 
        y_diagnosis.reshape(-1, 1),
        n_qubits=4,
        encoding='angle'
    )
    
    # Create hybrid model
    model = HybridModel(
        input_size=X_q.shape[1],
        n_qubits=4,
        quantum_depth=2,
        classical_layers=[16, 8]
    )
    
    return model, X_q, y_q
```

### 2. Hybrid Learning Tasks
Combining classical and quantum processing for complex pattern recognition.

```python
import torch.nn as nn
from quantum_ml.ml_integration import HybridModel

class HybridVisionModel(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        # Classical CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Quantum processing
        self.quantum = HybridModel(
            input_size=32 * 4 * 4,  # Flattened conv output
            n_qubits=n_qubits,
            quantum_depth=2,
            classical_layers=[64, 32]
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.quantum(x)
```

### 3. Quantum Circuit Optimization
Experiment with different quantum architectures and circuit designs.

```python
from quantum_ml.quantum_circuit import QuantumLayer
from quantum_ml.utils import visualize_quantum_state

def optimize_circuit_depth():
    # Test different circuit depths
    depths = [1, 2, 4, 8]
    circuits = []
    
    for depth in depths:
        # Create circuit with specific depth
        circuit = QuantumLayer(
            n_qubits=4,
            depth=depth  # Vary complexity
        )
        circuits.append(circuit)
        
        # Visualize circuit
        visualize_quantum_state(circuit.get_circuit())
        
    return circuits
```

### 4. Data Encoding Research
Test and compare different quantum encoding strategies.

```python
from quantum_ml.utils import encode_classical_data
import numpy as np

def compare_encodings(data):
    # Try different encoding methods
    encodings = {
        'angle': encode_classical_data(data, encoding='angle'),
        'amplitude': encode_classical_data(data, encoding='amplitude'),
        'phase': encode_classical_data(data, encoding='phase')
    }
    
    # Compare encoding properties
    for name, encoded in encodings.items():
        print(f"{name} encoding stats:")
        print(f"Mean: {np.mean(encoded):.4f}")
        print(f"Std: {np.std(encoded):.4f}")
        
    return encodings
```

## Extending the Framework

### Custom Quantum Layers

Create specialized quantum circuits by extending the QuantumLayer class:

```python
from quantum_ml.quantum_circuit import QuantumLayer

class CustomQuantumLayer(QuantumLayer):
    def _build_circuit(self):
        super()._build_circuit()
        # Add custom quantum operations
        self.circuit.h(0)  # Add Hadamard gate
        self.circuit.cx(0, 1)  # Add CNOT gate
        
        # Add custom parameterized operations
        theta = Parameter('θ_custom')
        self.parameters.append(theta)
        self.circuit.rz(theta, 1)
```

### Custom Encodings

Implement new data encoding strategies:

```python
import numpy as np
from quantum_ml.utils import encode_classical_data

def custom_fourier_encoding(data, n_qubits):
    """Example custom encoding using Fourier features"""
    # Normalize data
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Create Fourier features
    frequencies = 2.0 * np.pi * np.arange(n_qubits)
    features = np.cos(frequencies * data_norm[:, np.newaxis])
    
    return features
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
