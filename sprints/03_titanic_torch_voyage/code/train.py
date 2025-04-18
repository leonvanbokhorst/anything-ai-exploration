import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import argparse
import time

# Import custom modules
from titanic_dataset import TitanicDataset
from titanic_model import TitanicNet


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculates binary classification accuracy."""
    # Apply sigmoid to logits and threshold at 0.5
    predicted = torch.sigmoid(outputs) >= 0.5
    correct = (predicted == labels.bool()).sum().item()
    total = labels.size(0)
    # Handle case where total is 0 to avoid division by zero
    return (correct / total) if total > 0 else 0.0


def train_model_cv(
    data_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_splits: int = 5,
    device_str: str = "cpu",
):
    """
    Trains and evaluates the TitanicNet model using K-Fold Cross-Validation.

    Args:
        data_path: Path to the processed training CSV file.
        epochs: Number of training epochs per fold.
        batch_size: Number of samples per batch.
        learning_rate: Learning rate for the optimizer.
        n_splits: Number of folds for cross-validation.
        device_str: Device to use ('cpu' or 'cuda').
    """
    overall_start_time = time.time()

    # --- 1. Setup Device ---
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # --- 2. Load Full Data ---
    print("Loading full dataset...")
    full_dataset = TitanicDataset(data_path)
    n_features = full_dataset.n_features
    # Need features and targets as numpy arrays for StratifiedKFold split
    X = full_dataset.X.numpy()  # Or load directly if preferred
    y = full_dataset.y.numpy().squeeze()  # Target needs to be 1D for StratifiedKFold

    # --- 3. K-Fold Cross-Validation Setup ---
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )  # Added random_state for reproducibility
    fold_results = {"val_loss": [], "val_acc": []}

    print(f"\n--- Starting {n_splits}-Fold Cross-Validation --- ")

    # --- 4. Loop Through Folds ---
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        fold_start_time = time.time()
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        # --- 4a. Create Datasets and DataLoaders for this Fold ---
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        print(
            f"  Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}"
        )

        # --- 4b. Initialize Model, Loss, Optimizer for this Fold ---
        print("  Initializing model for this fold...")
        model = TitanicNet(input_features=n_features).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # --- 4c. Training Loop for this Fold ---
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            processed_batches = 0

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_acc += calculate_accuracy(outputs, labels)
                processed_batches += 1

            # Optional: Print training progress within fold (can be verbose)
            # print(f'    Epoch [{epoch+1}/{epochs}] Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {running_acc/len(train_loader):.4f}')

        # --- 4d. Validation Loop for this Fold ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        fold_results["val_loss"].append(avg_val_loss)
        fold_results["val_acc"].append(avg_val_acc)
        fold_end_time = time.time()
        print(
            f"  Fold {fold+1} COMPLETE - Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f} (Time: {fold_end_time - fold_start_time:.2f}s)"
        )

    # --- 5. Calculate and Print Average Results ---
    avg_cv_loss = np.mean(fold_results["val_loss"])
    avg_cv_acc = np.mean(fold_results["val_acc"])
    std_cv_acc = np.std(fold_results["val_acc"])

    print("\n--- Cross-Validation Finished ---")
    print(f"Average Validation Loss across {n_splits} folds: {avg_cv_loss:.4f}")
    print(
        f"Average Validation Accuracy across {n_splits} folds: {avg_cv_acc:.4f} (+/- {std_cv_acc:.4f})"
    )
    overall_end_time = time.time()
    print(f"Total CV time: {overall_end_time - overall_start_time:.2f} seconds")

    # --- Train final TitanicNet on full dataset and save weights ---
    print("Training final TitanicNet on full dataset for saving...")
    # Load full training data for final model
    full_dataset = TitanicDataset(data_path)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    final_model = TitanicNet(input_features=full_dataset.n_features).to(device)
    criterion_full = nn.BCEWithLogitsLoss()
    optimizer_full = optim.Adam(final_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        final_model.train()
        for inputs, labels in full_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_full.zero_grad()
            outputs = final_model(inputs)
            loss = criterion_full(outputs, labels)
            loss.backward()
            optimizer_full.step()
    # Derive results directory and save model
    sprint_dir_full = os.path.dirname(os.path.dirname(data_path))
    results_dir_full = os.path.join(sprint_dir_full, "results")
    os.makedirs(results_dir_full, exist_ok=True)
    model_path = os.path.join(results_dir_full, "titanic_model.pth")
    torch.save(final_model.state_dict(), model_path)
    print(f"Saved final model weights to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Titanic Model with K-Fold CV")
    sprint_dir = "sprints/03_titanic_torch_voyage"
    default_data_path = os.path.join(sprint_dir, "data", "processed_train.csv")

    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the processed training data CSV file",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs per fold"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds for Stratified K-Fold CV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for training ('cpu' or 'cuda')",
    )

    args = parser.parse_args()

    train_model_cv(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_splits=args.n_splits,
        device_str=args.device,
    )
