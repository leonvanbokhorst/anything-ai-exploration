# Sprint 4: Attachment-Theory Visualization Demo - Visualizing Bonds via Segmentation & Diffusion

## Sprint Goal:

To implement, train, and evaluate a U-Net architecture for image segmentation on the Oxford-IIIT Pet dataset, achieving a competitive intersection-over-union (IoU) score and understanding best practices for segmentation tasks.

## Tasks / Learning Objectives:

1.  [ ] **Data Acquisition & Preparation:**
    - [ ] Download the Oxford-IIIT Pet dataset with segmentation masks via `torchvision.datasets.OxfordIIITPet`.
    - [ ] Implement data loading, preprocessing, and augmentation (random flips, scaling, normalization).
    - [ ] Create PyTorch `Dataset` and `DataLoader` for training, validation, and test splits in `data/`.

2.  [ ] **Model Implementation:**
    - [ ] Implement the U-Net architecture in `code/unet.py`.
    - [ ] Ensure modular design for encoder and decoder blocks, skip connections, and configurable input/output channels.

3.  [ ] **Training Pipeline:**
    - [ ] Write training script `code/train_unet.py` with loss function (e.g., `CrossEntropyLoss`), optimizer (e.g., `Adam`), and learning rate scheduler.
    - [ ] Integrate metrics logging (e.g., IoU, Dice coefficient) and checkpointing.

4.  [ ] **Train & Evaluate Segmentation:**
    - [ ] Train the U-Net model on the prepared data and tune hyperparameters (learning rate, batch size, network depth).
    - [ ] Evaluate performance on the validation set and plot training curves.
    - [x] Generate and save sample segmentation outputs in `results/segmentation_example.png`.

5.  [ ] **Documentation & Reflection for Segmentation:**
    - [ ] Document model design decisions, preprocessing steps, and results in `docs/`.
    - [ ] Update this README with final metrics, example mask overlays, and reflections.

6.  **Attachment-Theory Visualization Demo:**
    - [ ] Extract frames from a sample parent-infant interaction video in `data/`.
    - [ ] Apply the trained U-Net model to segment parent and infant in each frame.
    - [ ] Perform pose estimation (MediaPipe/OpenPose) to extract keypoints and compute social signals (e.g., torso distance, gaze overlap).
    - [x] Implement a temporal GRU model in `code/attachment_gru.py` to predict attachment states (secure/anxious/avoidant) from these signals.
    - [x] Build a diffusion-based visualizer in `code/diffusion_visualizer.py` that maps predicted states to artistic overlays.
    - [x] Save example visualizations and interactive demos in `results/attachment_demo_overlay.png`.
    - [x] Document the process in `docs/01_gru_attachment_pipeline.md`.

## Definition of Done:

- [ ] Oxford-IIIT Pet dataset is downloaded, preprocessed, and accessible.
- [ ] U-Net model (`code/unet.py`) is implemented and importable.
- [ ] Training script (`code/train_unet.py`) runs successfully and trains the model.
- [ ] Model achieves a validation IoU > 0.7 (or comparable baseline).
- [ ] Metrics, plots, and sample predictions are saved in `results/`.
- [ ] Documentation is complete and updated in `docs/`.
- [ ] Attachment-Theory demo pipeline implemented and functional.
- [ ] GRU model and diffusion visualizer produce sample outputs in `results/`.
- [ ] Demo documented in `docs/`.
