# GRU-based Attachment-State Prediction Pipeline

## Overview
This pipeline ingests a time-series of social-signal features (e.g., torso distance, gaze overlap) and uses a GRU-based classifier to predict one of three attachment states: **Secure**, **Anxious**, or **Avoidant**.

## Dependencies
- Python 3.8+
- PyTorch >=1.10
- TQDM (progress bars)
- NumPy

## 1. Data Preparation
1. Extract per-frame social signals into a feature vector (e.g., distances, angles).
2. Save each interaction clip as a `.pt` file with:
   ```python
   {
     'features': torch.Tensor of shape [seq_len, num_features],
     'label':    int  # 0=secure, 1=anxious, 2=avoidant
   }
   ```
3. Place all `.pt` files under `data/signals/`.

## 2. Core Components

### 2.1 SocialSignalDataset
```python
class SocialSignalDataset(Dataset):
    def __init__(self, signal_files: List[str]):
        self.signal_files = signal_files
    def __getitem__(self, idx):
        data = torch.load(self.signal_files[idx])
        return data['features'], data['label']
```
Wraps your `.pt` files into a PyTorch `Dataset` for easy batching.

### 2.2 AttachmentGRU
```python
model = AttachmentGRU(
    input_size=num_features,
    hidden_size=128,
    num_layers=2,
    num_classes=3,
    dropout=0.1
)
``` 
A simple GRU followed by a linear classifier on the last hidden state.

### 2.3 build_dataloader
```python
loader = build_dataloader(signal_files, batch_size=32, shuffle=True, num_workers=0)
``` 
Creates a `DataLoader` with batching and optional shuffling.

## 3. Training & Evaluation
Integrate with standard PyTorch loops:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    train_epoch(model, loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)
    print(f"Epoch {epoch}: Val Acc={val_acc:.2f}")
```  
Swap in `CrossEntropyLoss` since labels are integer classes.

## 4. Quickstart
From the `code/` directory:
```bash
python attachment_gru.py
```  
This will load any `.pt` files in `data/signals/`, build the model, and print a batch of predictions vs. labels.

## 5. Next Steps & Tips
- **Feature Engineering:** Add motion dynamics or facial keypoint metrics.
- **Hyperparameter Tuning:** Experiment with `hidden_size`, `num_layers`, and learning rate.
- **Model Variants:** Try LSTM or Transformer encoders for richer temporal modeling.
- **Evaluation:** Track accuracy, confusion matrix, and ROC curves.

---
*Last updated: `$(date '+%Y-%m-%d')`*

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
