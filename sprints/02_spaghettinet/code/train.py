import os
import sys

# Allow import of spaghettinet module in this directory
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spaghettinet import SpaghettiNet


def train(model: nn.Module, device: torch.device, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int) -> None:
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(loader.dataset)}]	Loss: {loss.item():.6f}")
    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.6f}\n")


def test(model: nn.Module, device: torch.device, loader: DataLoader, criterion: nn.Module) -> None:
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"Test set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n")


def main() -> None:
    # Configurations
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    lr = 1e-3
    sparsity = 0.7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Data transforms and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Model, optimizer, loss
    model = SpaghettiNet(28*28, 128, 10, sparsity=sparsity).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Run training and testing
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

    # Save the final model
    result_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'spaghettinet_mnist.pt')
    torch.save(model.state_dict(), result_path)
    print(f"Training complete. Model saved to {result_path}")


if __name__ == "__main__":
    main() 