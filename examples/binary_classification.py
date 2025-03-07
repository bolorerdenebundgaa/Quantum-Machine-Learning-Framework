"""
Binary Classification Example
===========================

Demonstrates using the QuaNTUM ML framework for binary classification.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantum_ml.ml_integration import HybridModel, QuantumOptimizer
from quantum_ml.utils import create_quantum_dataset, calculate_quantum_metrics
from quantum_ml.config import FrameworkConfig

def main():
    # Generate synthetic dataset
    X, y = make_circles(n_samples=200, noise=0.1, factor=0.3)
    X = StandardScaler().fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize framework configuration
    config = FrameworkConfig(
        quantum_backend={
            'name': 'qasm_simulator',
            'n_qubits': 4,
            'shots': 1000
        },
        ml_config={
            'batch_size': 32,
            'learning_rate': 0.01,
            'n_epochs': 20
        }
    )
    
    # Create quantum datasets
    X_train_q, y_train_q = create_quantum_dataset(
        X_train, y_train.reshape(-1, 1),
        n_qubits=config.quantum_backend.n_qubits,
        encoding='angle'
    )
    X_test_q, y_test_q = create_quantum_dataset(
        X_test, y_test.reshape(-1, 1),
        n_qubits=config.quantum_backend.n_qubits,
        encoding='angle'
    )
    
    # Initialize hybrid model
    model = HybridModel(
        input_size=X_train_q.shape[1],
        n_qubits=config.quantum_backend.n_qubits,
        quantum_depth=2,
        classical_layers=[16, 8]
    )
    
    # Setup training
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.ml_config.learning_rate
    )
    quantum_optimizer = QuantumOptimizer(model, loss_fn, optimizer)
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.ml_config.n_epochs):
        total_loss = 0
        n_batches = 0
        
        # Create batches
        permutation = torch.randperm(X_train_q.size()[0])
        for i in range(0, X_train_q.size()[0], config.ml_config.batch_size):
            indices = permutation[i:i + config.ml_config.batch_size]
            batch_x, batch_y = X_train_q[indices], y_train_q[indices]
            
            # Optimization step
            _, loss = quantum_optimizer.step(batch_x, batch_y)
            total_loss += loss.item()
            n_batches += 1
            
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{config.ml_config.n_epochs}, "
              f"Loss: {avg_loss:.4f}")
    
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test_q)).numpy()
        
    # Calculate metrics
    metrics = calculate_quantum_metrics(y_pred, y_test_q.numpy())
    print("\nTest Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()
