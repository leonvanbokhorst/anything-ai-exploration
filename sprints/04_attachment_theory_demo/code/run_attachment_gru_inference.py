import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from attachment_gru import AttachmentGRU, build_dataloader

LABEL_NAMES = ['secure', 'anxious', 'avoidant']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script for trained AttachmentGRU model'
    )
    parser.add_argument(
        '--signals-dir', type=str, default='data/signals',
        help='Directory containing .pt social-signal files'
    )
    parser.add_argument(
        '--model-path', type=str, default='results/best_model.pt',
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for inference'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Gather signal files
    pattern = os.path.join(args.signals_dir, '*.pt')
    signal_files = sorted(glob.glob(pattern))
    if not signal_files:
        print(f"No .pt files found in {args.signals_dir}")
        return

    # Determine input size from first sample
    sample = torch.load(signal_files[0])
    input_size = sample['features'].shape[1]

    # Build dataloader
    loader: DataLoader = build_dataloader(
        signal_files, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Load model
    model = AttachmentGRU(input_size=input_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    print("Running inference on social-signal sequences...")
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            for pred, true in zip(preds, labels.tolist()):
                print(f"Predicted: {LABEL_NAMES[pred]}, True: {LABEL_NAMES[true]}")


if __name__ == '__main__':
    main() 