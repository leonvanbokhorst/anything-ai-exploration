"""
Temporal GRU model to predict attachment states (secure, anxious, avoidant) from social signals.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SocialSignalDataset(Dataset):
    """
    Dataset for loading sequences of social signals and their labels.
    Each file in `signal_files` should contain a dict with keys 'features' (Tensor of shape [seq_len, num_features])
    and 'label' (int in {0,1,2}).
    """

    def __init__(self, signal_files: List[str]):
        self.signal_files = signal_files

    def __len__(self) -> int:
        return len(self.signal_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data = torch.load(self.signal_files[idx])
        features = data["features"]  # Tensor shape [seq_len, num_features]
        label = int(data["label"])  # Attachment state: 0=secure,1=anxious,2=avoidant
        return features, label


class AttachmentGRU(nn.Module):
    """
    GRU-based classifier for attachment states.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_size]
        _, hn = self.gru(x)  # hn: [num_layers, batch_size, hidden_size]
        last_hidden = hn[-1]  # [batch_size, hidden_size]
        logits = self.classifier(last_hidden)
        return logits  # [batch_size, num_classes]


def build_dataloader(
    signal_files: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Utility to create a DataLoader for SocialSignalDataset.
    """
    dataset = SocialSignalDataset(signal_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return loader


if __name__ == "__main__":
    import glob

    # Example usage
    seq_files = glob.glob("data/signals/*.pt")
    dataloader = build_dataloader(seq_files)
    # replace 10 with the actual number of social-signal features
    seq_len_features = 10
    model = AttachmentGRU(input_size=seq_len_features)
    for features, labels in dataloader:
        logits = model(features)  # [batch, num_classes]
        preds = torch.argmax(logits, dim=1)
        print(f"Batch preds={preds.tolist()}, labels={labels}")
        break
