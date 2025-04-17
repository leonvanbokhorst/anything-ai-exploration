import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpaghettiLayer(nn.Module):
    """
    A neural layer with random, spaghetti-like sparse connectivity.
    Mimics tangled spaghetti by applying a frozen random mask to weights.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.5,
        bias: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__()
        if not 0 <= sparsity < 1:
            raise ValueError(f"Sparsity must be in [0,1), got {sparsity}")
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        # Optionally fix randomness for reproducibility
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Initialize raw weight parameter with shape (out_features, in_features)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Create boolean mask for connections: True = active, same shape as weight
        mask = torch.rand(out_features, in_features) > sparsity
        self.register_buffer("connectivity_mask", mask.float())

        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Apply the spaghetti mask
        with torch.no_grad():
            self.weight.data.mul_(self.connectivity_mask)

        # Initialize bias
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applies a linear transform with sparsely masked weights.
        """
        effective_weight = self.weight * self.connectivity_mask
        return F.linear(input, effective_weight, self.bias)


class SpaghettiNet(nn.Module):
    """
    A minimal feedforward network using two SpaghettiLayer instances.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        sparsity: float = 0.5,
    ):
        super().__init__()
        self.layer1 = SpaghettiLayer(input_dim, hidden_dim, sparsity)
        self.activation = nn.ReLU()
        self.layer2 = SpaghettiLayer(hidden_dim, output_dim, sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)


if __name__ == "__main__":
    # Smoke test for dimensions
    batch_size = 8
    input_dim = 16
    hidden_dim = 32
    output_dim = 10
    model = SpaghettiNet(input_dim, hidden_dim, output_dim, sparsity=0.7)
    dummy_input = torch.randn(batch_size, input_dim)
    out = model(dummy_input)
    print(f"Output shape: {tuple(out.shape)}") 