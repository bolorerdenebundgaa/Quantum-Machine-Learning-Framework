"""
Medical Diagnosis Example
========================

Demonstrates using the QuaNTUM ML Framework for medical diagnosis classification.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantum_ml.ml_integration import HybridModel, QuantumOptimizer
from quantum_ml.utils import create_quantum_dataset, calculate_quantum_metrics
from quantum_ml.config import FrameworkConfig

def generate_synthetic_medical_data(n_samples=1000, n_features=10):
    """Generate synthetic medical data for demonstration."""
    # Generate feature data (e.g., blood tests, vital signs, etc.)
    X = np.random.randn(n_samples, n_features)
    
    # Generate diagnosis based on complex feature interactions
    y = np.zeros(n_samples)
    
    # Simulate some medical conditions:
    # - High values in first two features
    # - Correlation between features 3 and 4
    # - Threshold on feature 5
    condition1 = (X[:, 0] > 0.5) & (X[:, 1] > 0.5)
    condition2 = X[:, 3] * X[:, 4] > 0
    condition3 = X[:, 5] > 1.0
    
    y = (condition1 & condition2) | condition3
    y = y.astype(int)
    
    return X, y

def main():
    # Generate synthetic medical data
    print("Generating synthetic medical data...")
    X, y = generate_synthetic_medical_data()
    
    # Preprocess data
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
    print("Encoding data into quantum states...")
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
    print("Creating hybrid quantum-classical model...")
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
    print("\nTraining model...")
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
    
    # Example prediction
    print("\nExample Predictions:")
    for i in range(5):
        true_label = y_test[i]
        pred_prob = y_pred[i][0]
        diagnosis = "Positive" if pred_prob > 0.5 else "Negative"
        print(f"Patient {i + 1}:")
        print(f"True Label: {true_label}")
        print(f"Predicted Probability: {pred_prob:.4f}")
        print(f"Diagnosis: {diagnosis}\n")

if __name__ == '__main__':
    main()
