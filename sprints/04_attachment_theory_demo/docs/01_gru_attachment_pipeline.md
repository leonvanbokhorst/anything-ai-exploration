# GRU-based Attachment-State Prediction Pipeline

Welcome, novice AI explorer! This document walks you through how we use a simple GRU (Gated Recurrent Unit) to turn a sequence of social signals into an attachment-state prediction (secure, anxious, or avoidant).

---

## 1. What is a GRU?

A GRU is a type of **recurrent neural network** (RNN) that processes sequences step by step. Unlike standard feed-forward networks, GRUs keep an internal memory (`hidden state`) that captures information from previous time steps. This makes GRUs well-suited for time-series or sequence tasks—like interpreting how social cues evolve over a video.

Key terms:

- **Sequence**: an ordered list of feature vectors (e.g., distances, angles over time).
- **Hidden state**: the GRU's internal memory at each step.
- **Output logits**: raw scores for each attachment state before converting to probabilities.

---

## 2. Preparing Your Data

1. **Extract Social Signals**: From each video frame, compute a small list of numbers (features) such as:

   - Distance between parent and infant bodies.
   - Overlap of head/gaze directions.
   - Any other social metric you design.

2. **Save as `.pt` Files**: Each video clip's signals and label go into a single Torch file. The file should be a Python dict with:

   ```python
   {
       'features': Tensor of shape [sequence_length, num_features],
       'label':    Integer (0=secure, 1=anxious, 2=avoidant)
   }
   ```

3. **Organize**: Put all `.pt` files under `data/signals/` so our code can load them automatically.

---

## 3. Key Python Components

### 3.1 SocialSignalDataset

Located in `code/attachment_gru.py`. This class:

- Inherits `torch.utils.data.Dataset`.
- Reads each `.pt` file.
- Returns a tuple `(features, label)` for training or inference.

### 3.2 AttachmentGRU

Also in `code/attachment_gru.py`. This is our model:

- **GRU Layers**: reads sequences of feature vectors.
- **Classifier**: a final `Linear` layer turns the last hidden state into 3 output scores.

Constructor signature:

```python
AttachmentGRU(
    input_size:  int,    # Number of features per time step
    hidden_size: int=128,# Size of the GRU's hidden state
    num_layers:  int=2,  # How many stacked GRU layers
    num_classes: int=3,  # We have three attachment states
    dropout:     float=0.1
)
```

### 3.3 build_dataloader

A helper function that wraps `SocialSignalDataset` in a `DataLoader`. It handles batching, shuffling, and parallel loading.

---

## 4. Running the Script

1. **Install dependencies** (if you haven't already):

   ```bash
   pip install torch torchvision
   ```

2. **Place your `.pt` files** in `data/signals/`.

3. **Run the example**:

   ```bash
   python code/attachment_gru.py
   ```

   The script will:

   - Load a batch of signals!
   - Pass them through the GRU model.
   - Print out predicted vs. true labels for the first batch.

4. **Adjust Parameters**:
   - Change `seq_len_features` in the `__main__` block to match your feature dimension.
   - Tweak batch size, hidden size, or number of layers directly in the code.

---

## 5. Next Steps & Tips

- **Training Loop**: You can integrate this model into a full training loop with optimizers and loss functions (e.g., `CrossEntropyLoss`).
- **Evaluation**: Track accuracy or confusion matrices to see which states the model confuses.
- **Feature Engineering**: Better social signals (like gaze vectors or motion speed) often yield better classification.
- **Debugging**: Start with small synthetic sequences to verify your pipeline before using real videos.

Happy modeling, and may your lab experiments never crash! 🧪
