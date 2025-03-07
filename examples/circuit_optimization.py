"""
Circuit Optimization Example
=========================

Demonstrates circuit optimization techniques and encoding comparison
using the QuaNTUM ML Framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from quantum_ml.quantum_circuit import QuantumLayer
from quantum_ml.utils import (
    encode_classical_data,
    visualize_quantum_state,
    create_quantum_dataset
)
from quantum_ml.config import FrameworkConfig

def compare_circuit_depths(X, y, depths=[1, 2, 4, 8], n_qubits=4):
    """Compare different circuit depths."""
    results = {}
    
    for depth in depths:
        print(f"\nTesting circuit depth: {depth}")
        
        # Create circuit
        circuit = QuantumLayer(n_qubits=n_qubits, depth=depth)
        
        # Visualize circuit
        print(f"Circuit with depth {depth}:")
        visualize_quantum_state(circuit.get_circuit())
        
        # Test with some random parameters
        n_params = len(circuit.parameters)
        test_params = np.random.uniform(0, 2*np.pi, n_params)
        output = circuit.forward(test_params)
        
        results[depth] = {
            'n_parameters': n_params,
            'circuit': circuit,
            'test_output': output
        }
        
        print(f"Number of parameters: {n_params}")
        print(f"Test output shape: {output.shape}")
    
    return results

def compare_encoding_methods(X, encodings=['angle', 'amplitude', 'phase']):
    """Compare different encoding methods."""
    results = {}
    
    for encoding in encodings:
        print(f"\nTesting {encoding} encoding:")
        encoded_data = encode_classical_data(X, encoding=encoding)
        
        # Calculate statistics
        stats = {
            'mean': np.mean(encoded_data),
            'std': np.std(encoded_data),
            'min': np.min(encoded_data),
            'max': np.max(encoded_data)
        }
        
        results[encoding] = {
            'encoded_data': encoded_data,
            'stats': stats
        }
        
        print("Statistics:")
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value:.4f}")
    
    return results

def plot_encoding_comparison(results):
    """Plot comparison of different encodings."""
    n_encodings = len(results)
    fig, axes = plt.subplots(1, n_encodings, figsize=(5*n_encodings, 4))
    
    for i, (encoding, data) in enumerate(results.items()):
        ax = axes[i] if n_encodings > 1 else axes
        encoded = data['encoded_data']
        
        if np.iscomplexobj(encoded):
            # For complex data, plot magnitude
            values = np.abs(encoded.flatten())
            title_suffix = " (Magnitude)"
        else:
            values = encoded.flatten()
            title_suffix = ""
        
        # Plot histogram of values
        ax.hist(values, bins=50)
        ax.set_title(f"{encoding.capitalize()} Encoding{title_suffix}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def plot_circuit_complexity(results):
    """Plot circuit complexity metrics."""
    depths = list(results.keys())
    n_params = [data['n_parameters'] for data in results.values()]
    
    plt.figure(figsize=(8, 4))
    plt.plot(depths, n_params, 'bo-')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Number of Parameters')
    plt.title('Circuit Complexity vs Depth')
    plt.grid(True)
    plt.show()

def main():
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    X, y = make_moons(n_samples=1000, noise=0.1)
    X = StandardScaler().fit_transform(X)
    
    # Compare circuit depths
    print("\nComparing different circuit depths...")
    circuit_results = compare_circuit_depths(X, y)
    
    # Plot circuit complexity
    print("\nPlotting circuit complexity...")
    plot_circuit_complexity(circuit_results)
    
    # Compare encoding methods
    print("\nComparing different encoding methods...")
    encoding_results = compare_encoding_methods(X)
    
    # Plot encoding comparison
    print("\nPlotting encoding comparison...")
    plot_encoding_comparison(encoding_results)
    
    # Detailed encoding analysis
    print("\nDetailed encoding analysis:")
    for encoding, results in encoding_results.items():
        print(f"\n{encoding.capitalize()} Encoding Analysis:")
        stats = results['stats']
        print("Distribution Statistics:")
        for stat_name, value in stats.items():
            print(f"- {stat_name}: {value:.4f}")
        
        encoded_data = results['encoded_data']
        print("Shape Analysis:")
        print(f"- Input shape: {X.shape}")
        print(f"- Encoded shape: {encoded_data.shape}")
        
        # Calculate additional metrics
        print("Additional Metrics:")
        print(f"- Sparsity: {np.mean(np.abs(encoded_data) < 1e-6):.4f}")
        print(f"- Dynamic Range: {np.max(encoded_data) - np.min(encoded_data):.4f}")

if __name__ == '__main__':
    main()
