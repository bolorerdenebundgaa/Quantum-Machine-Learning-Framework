"""
Hybrid Vision Example
===================

Demonstrates using the QuaNTUM ML Framework for image classification
by combining classical CNN with quantum processing.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from quantum_ml.ml_integration import HybridModel
from quantum_ml.config import FrameworkConfig

class HybridVisionModel(nn.Module):
    """Hybrid model combining CNN with quantum processing."""
    
    def __init__(self, n_qubits=4, n_classes=10):
        super().__init__()
        # Classical CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Quantum processing
        self.quantum = HybridModel(
            input_size=64 * 4 * 4,  # Flattened conv output
            n_qubits=n_qubits,
            quantum_depth=2,
            classical_layers=[256, 64]
        )
        
        # Final classification layer
        self.classifier = nn.Linear(1, n_classes)
        
    def forward(self, x):
        # Classical feature extraction
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Quantum processing
        x = self.quantum(x)
        
        # Classification
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch: {batch_idx + 1}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * correct / total:.2f}%")
    
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    # Configuration
    config = FrameworkConfig()
    device = torch.device(config.device)
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download and load CIFAR10 dataset
    print("Downloading/Loading CIFAR10 dataset...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    trainset = CIFAR10(root=data_dir, train=True, download=True,
                       transform=transform)
    testset = CIFAR10(root=data_dir, train=False, download=True,
                      transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True,
                           num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False,
                          num_workers=2)
    
    # Create model
    print("Creating hybrid vision model...")
    model = HybridVisionModel(n_qubits=4, n_classes=10).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 5
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device
        )
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"Testing  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    print("\nTraining completed!")

if __name__ == '__main__':
    main()
