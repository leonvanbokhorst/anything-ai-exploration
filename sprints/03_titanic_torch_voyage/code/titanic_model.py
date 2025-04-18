import torch
import torch.nn as nn
import torch.nn.functional as F


class TitanicNet(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for the Titanic dataset."""

    def __init__(self, input_features: int):
        """
        Initializes the layers of the network.

        Args:
            input_features: The number of input features (columns in the preprocessed data).
        """
        super(TitanicNet, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(
            input_features, 32
        )  # Input features -> Hidden Layer 1 (32 neurons)
        self.fc2 = nn.Linear(32, 16)  # Hidden Layer 1 -> Hidden Layer 2 (16 neurons)
        self.fc3 = nn.Linear(16, 1)  # Hidden Layer 2 -> Output Layer (1 neuron)

        # Optional: Add Batch Normalization or Dropout later if needed
        # self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.4)  # Adding dropout

        print(f"Initialized TitanicNet with {input_features} input features.")
        print(
            "Architecture: Linear(in={input_features}, out=32) -> ReLU -> Dropout(p=0.4) -> Linear(in=32, out=16) -> ReLU -> Dropout(p=0.4) -> Linear(in=16, out=1)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x: The input tensor (batch_size, input_features).

        Returns:
            The output tensor (batch_size, 1), representing the raw logits.
        """
        # Apply layers with ReLU activation functions and Dropout
        x = F.relu(self.fc1(x))
        # x = self.bn1(x) # Optional Batch Normalization
        # x = self.dropout(x) # Optional Dropout
        x = self.dropout(x)  # Apply dropout after first ReLU
        x = F.relu(self.fc2(x))
        # x = self.dropout(x) # Optional Dropout
        x = self.dropout(x)  # Apply dropout after second ReLU
        x = self.fc3(x)  # Output raw logits
        return x


# Example usage (if run directly)
if __name__ == "__main__":
    num_input_features = 17  # As determined from processed_train.csv
    print(f"Creating an example TitanicNet with {num_input_features} features...")
    model = TitanicNet(input_features=num_input_features)
    print("Model created successfully.")

    # Create a dummy input tensor (batch of 2 samples, 17 features each)
    dummy_input = torch.randn(2, num_input_features)
    print(f"\nTesting forward pass with dummy input of shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape}")
    print(f"Example output (logits):\n{output}")
