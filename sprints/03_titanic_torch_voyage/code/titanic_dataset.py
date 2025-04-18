import torch
import pandas as pd
from torch.utils.data import Dataset
import os


class TitanicDataset(Dataset):
    """PyTorch Dataset for loading the preprocessed Titanic data."""

    def __init__(self, csv_path: str):
        """
        Initializes the dataset by loading data from a CSV file.

        Args:
            csv_path: Path to the preprocessed CSV file.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Separate features (X) and target (y)
        if "Survived" in df.columns:
            print("  Found 'Survived' column (target).")
            # Ensure target is float for BCEWithLogitsLoss
            self.y = torch.tensor(df["Survived"].values, dtype=torch.float32).unsqueeze(
                1
            )  # Target shape: (n_samples, 1)
            self.X = torch.tensor(
                df.drop("Survived", axis=1).values, dtype=torch.float32
            )
            self.has_target = True
        else:
            print("  No 'Survived' column found (assuming test set).")
            self.y = None  # No target for test set
            self.X = torch.tensor(df.values, dtype=torch.float32)
            self.has_target = False

        self.n_samples, self.n_features = self.X.shape
        print(f"  Loaded {self.n_samples} samples with {self.n_features} features.")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Retrieves a single sample (features and optionally target) from the dataset.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            If target exists: A tuple containing (features, target) tensors.
            If target doesn't exist: Only the features tensor.
        """
        if self.has_target:
            return self.X[index], self.y[index]
        else:
            return self.X[index]  # Only return features for test set


# Example usage (if run directly)
if __name__ == "__main__":
    # Assume the script is run from the workspace root
    sprint_dir = "sprints/03_titanic_torch_voyage"
    train_data_path = os.path.join(sprint_dir, "data", "processed_train.csv")
    test_data_path = os.path.join(sprint_dir, "data", "processed_test.csv")

    print("\n--- Testing Training Dataset ---")
    try:
        train_dataset = TitanicDataset(train_data_path)
        print(f"Number of training samples: {len(train_dataset)}")
        features, target = train_dataset[0]  # Get first sample
        print(
            f"First training sample - Features shape: {features.shape}, Target shape: {target.shape}"
        )
        print(
            f"First training sample - Features dtype: {features.dtype}, Target dtype: {target.dtype}"
        )
    except FileNotFoundError as e:
        print(e)
        print("Please ensure preprocess_data.py has been run successfully.")

    print("\n--- Testing Test Dataset ---")
    try:
        test_dataset = TitanicDataset(test_data_path)
        print(f"Number of test samples: {len(test_dataset)}")
        features = test_dataset[0]  # Get first sample
        print(f"First test sample - Features shape: {features.shape}")
        print(f"First test sample - Features dtype: {features.dtype}")
    except FileNotFoundError as e:
        print(e)
        print("Please ensure preprocess_data.py has been run successfully.")
