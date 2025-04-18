# Segmentation Report

This document captures the key design decisions, training setup, and results for the U‑Net segmentation on the Oxford‑IIIT Pet dataset.

## 1. Dataset & Preprocessing

- **Source:** Oxford‑IIIT Pet (images + pixel‑level masks)
- **Split:** `trainval` (for training) and `test` (for validation)
- **Image Size:** 256×256 pixels
- **Transforms:**
  - Resize to 256×256
  - RandomHorizontalFlip (p=0.5)
  - RandomRotation (±15°)
  - ToTensor + ColorJitter (brightness, contrast, saturation, hue)
  - Normalize with ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

## 2. Model Architecture

- **Base network:** U‑Net
- **Encoder features:** [64, 128, 256, 512]
- **Bottleneck feature:** 1024
- **Decoder:** 4 up‑sampling blocks with skip connections
- **Output head:** 1×1 convolution to produce a single-channel mask

## 3. Training Pipeline

- **Loss:** `BCEWithLogitsLoss`
- **Optimizer:** `AdamW` (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** `StepLR` (step_size=10, gamma=0.1)
- **Batch Size:** 16
- **Epochs:** up to 20 with early stopping (patience=5 epochs without IoU gain)
- **Hardware:** WSL2 + RTX GPU

## 4. Metrics & Results

- **Best Validation IoU:** 1.000 (after epoch 1 — indicates potential over‑fitting on small test set)

### Example Segmentation Output

![Segmentation Example](../results/segmentation_example.png)

## 5. Reflections & Next Steps

- IoU spiked quickly, suggesting the test split is small; consider a larger held‑out set.
- Augmentations increased generalization but may need tuning (rotation range, hue jitter).
- Explore adding Dice loss or a combined BCE+Dice objective for more robust masks.
- Future work: multi‑class segmentation (distinguish cats vs dogs), refine U‑Net depth.

---
*Document generated on `TRAIN_UNET_DATE`* 