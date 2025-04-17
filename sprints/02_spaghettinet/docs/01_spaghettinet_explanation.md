# 01: Spaghettinet Connectivity Explanation

## 🎲 Introduction
Spaghettinet is a whimsical neural network layer inspired by a tangled bowl of spaghetti. Instead of dense connections between neurons, it uses a frozen, random sparse mask—just like randomly draped noodles—to explore whether unusual wiring can still learn patterns efficiently.

## 🍝 Core Concept
- A **SpaghettiLayer** acts like a standard linear (fully connected) layer but applies a fixed binary mask (`connectivity_mask`) to its weight matrix.
- The mask has shape `(out_features, in_features)` and keeps only a fraction of connections alive.
- **Sparsity** controls the percentage of dropped (masked) connections: `sparsity=0.7` means 70% of weights are zeroed out.

## 🛠 Implementation Details
- **Class:** `SpaghettiLayer(nn.Module)`
- **Key Parameters:**
  - `in_features`: Number of incoming neurons (e.g., 784 for 28×28 MNIST).
  - `out_features`: Number of outgoing neurons (e.g., 128 hidden units).
  - `sparsity`: Fraction of connections to drop (0 ≤ sparsity < 1).
  - `random_seed`: Optional seed for reproducible wiring.

```python
from spaghettinet import SpaghettiLayer

# Create a layer with 784→128 dims and 70% sparsity
layer = SpaghettiLayer(
    in_features=784,
    out_features=128,
    sparsity=0.7,
    random_seed=42
)
```

After initialization, weights are Kaiming‐initialized, then elementwise-multiplied by the binary mask to lock in the random spaghetti wiring.

## 🚂 Training with Spaghettinet
The `train.py` script in `code/` integrates the layer into a small feedforward network and trains on MNIST:
- Uses `SpaghettiNet`, a two-layer model (SpaghettiLayer→ReLU→SpaghettiLayer).
- Achieved ~96.8% test accuracy with `sparsity=0.7` after 5 epochs.

## 🔍 Hyperparameter Exploration
- **Sparsity sweep:** Try `0.1, 0.5, 0.9` to see how much wiring you can remove before performance collapses.
- **Layer dims:** Experiment with different hidden sizes (64, 256, …).
- **Random seeds:** Rerun with different seeds to inspect variance in connectivity patterns.

## 🏁 Next Steps
1. **Quantify sparsity vs. accuracy**: plot a curve of sparsity vs. test accuracy.
2. **Entropy measure**: compute graph entropy or connectivity distribution.
3. **Visual reports**: save PNGs of noodle graphs and training curves in `results/`.

Happy noodling! 🍜 