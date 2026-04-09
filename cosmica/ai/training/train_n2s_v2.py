"""
Noise2Self Training Script V2.
- Loads strictly split data (train.dat, val.dat).
- Saves checkpoint every epoch.
- Lower LR, Higher Weight Decay.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cosmica.ai.models.unet import UNet

log = logging.getLogger(__name__)

# Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 3e-4  # Slower, more stable
WEIGHT_DECAY = 1e-3   # Stronger regularization to prevent overfitting
MASK_RATIO = 0.20     # Harder task
NUM_WORKERS = 0  # 0 workers to avoid RAM issues with memmap

class MemmapDataset(Dataset):
    def __init__(self, memmap_path: Path, meta_path: Path, augmentation: bool = True):
        with open(meta_path) as f:
            total = int(f.read().strip())
        self.data = np.memmap(memmap_path, dtype=np.float32, mode='r',
                              shape=(total, PATCH_SIZE, PATCH_SIZE))
        self.augmentation = augmentation

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        patch = self.data[idx]
        if self.augmentation:
            if np.random.random() > 0.5: patch = np.flip(patch, axis=0).copy()
            if np.random.random() > 0.5: patch = np.flip(patch, axis=1).copy()
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k).copy()
            jitter = np.random.uniform(0.8, 1.2)
            patch = np.clip(patch * jitter, 0, 1).astype(np.float32)
        return torch.from_numpy(patch).unsqueeze(0)

def create_masked_input(patches: torch.Tensor):
    n, c, h, w = patches.shape
    mask = torch.rand(n, c, h, w, device=patches.device) < MASK_RATIO
    target = patches.clone()
    masked = patches.clone()
    masked[mask] = 0.0
    return masked, target, mask

def train():
    output_dir = Path("cosmica/ai/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("training_data")
    train_dat = data_dir / "train.dat"
    val_dat = data_dir / "val.dat"
    
    if not train_dat.exists():
        log.error("Data not found! Run prepare_data_v2.py first.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Model: base=32, depth=4 (7.7M params)
    model = UNet(in_channels=1, out_channels=1, base_features=32, depth=4).to(device)
    log.info(f"Model Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Datasets
    train_dataset = MemmapDataset(train_dat, data_dir / "train_meta.txt", augmentation=True)
    val_dataset = MemmapDataset(val_dat, data_dir / "val_meta.txt", augmentation=False)
    
    log.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val patches")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            masked, target, mask = create_masked_input(batch)
            pred = model(masked)
            loss = ((pred - target) ** 2)[mask].mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Val
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                masked, target, mask = create_masked_input(batch)
                pred = model(masked)
                loss = ((pred - target) ** 2)[mask].mean()
                val_loss += loss.item()
                n_val += 1
        
        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(n_val, 1)
        
        log.info(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.0e}")

        # CHECKPOINT EVERY EPOCH
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val,
            "config": {"in_channels": 1, "out_channels": 1, "base_features": 32, "depth": 4}
        }, ckpt_path)
        log.info(f" -> Saved checkpoint: {ckpt_path}")

    log.info("TRAINING COMPLETE.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    train()
