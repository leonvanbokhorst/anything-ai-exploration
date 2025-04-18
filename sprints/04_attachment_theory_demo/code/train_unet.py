import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.transforms import PILToTensor
from PIL import Image
import numpy as np
from tqdm import tqdm

from unet import UNet


class PetSegmentationDataset(Dataset):
    """
    Dataset wrapper for Oxford-IIIT Pet segmentation masks.
    """
    def __init__(
        self,
        root: str,
        split: str = 'trainval',
        img_size: int = 256
    ) -> None:
        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types='segmentation',
            download=False,
        )
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),  # augmentation
            transforms.RandomRotation(15),      # augmentation
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # augmentation on tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            PILToTensor(),
            lambda x: x.squeeze(0).long(),  # convert to [H, W] tensor
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[idx]  # PIL.Image, PIL.Image
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        # binary mask: pet vs background
        mask = (mask > 0).float()
        return image, mask


# Average per-sample IoU over a batch
def iou_metric(
    preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6
) -> float:
    # preds and labels: [N, H, W] in [0,1]
    preds_bin = (preds > 0.5).float()
    batch_size = preds_bin.shape[0]
    preds_flat = preds_bin.view(batch_size, -1)
    labels_flat = labels.view(batch_size, -1)
    intersection = (preds_flat * labels_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + labels_flat.sum(dim=1) - intersection
    ious = (intersection + eps) / (union + eps)
    return ious.mean().item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc='Training'):  # type: ignore
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)  # [N, H, W]
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)  # type: ignore


# Validate and compute average per-sample IoU
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    val_iou_total = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Validation'):  # type: ignore
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, masks)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(logits)
            batch_iou = iou_metric(preds, masks)
            val_iou_total += batch_iou * imgs.size(0)
    avg_iou = val_iou_total / len(loader.dataset)
    return val_loss / len(loader.dataset), avg_iou


def main() -> None:
    parser = argparse.ArgumentParser(description='Train U-Net for pet segmentation')
    parser.add_argument(
        '--data-root', type=str, default='data', help='dataset root directory'
    )
    parser.add_argument('--epochs', type=int, default=20, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--img-size', type=int, default=256, help='resized image size')
    parser.add_argument(
        '--save-dir', type=str, default='results', help='checkpoint & metrics dir'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = PetSegmentationDataset(args.data_root, split='trainval', img_size=args.img_size)
    val_ds = PetSegmentationDataset(args.data_root, split='test', img_size=args.img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # disable multiprocessing for WSL2
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # disable multiprocessing for WSL2
        pin_memory=True,
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Early stopping parameters
    best_iou = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val IoU: {val_iou:.4f}"
        )

        # Save checkpoints
        epoch_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), epoch_path)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
        else:
            no_improve += 1
        # Early stopping check
        if no_improve >= patience:
            print(f"No improvement in IoU for {patience} epochs—stopping early.")
            break


if __name__ == '__main__':
    main() 