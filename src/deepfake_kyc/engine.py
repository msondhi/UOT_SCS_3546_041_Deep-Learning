"""Training and inference loops."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def warmup_cosine(warmup_steps: int, total_steps: int):
    """LR multiplier: linear warm-up, then cosine decay to zero (for ``LambdaLR``)."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def train_one_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    device,
    freeze_batch_norm: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """One pass over the training data. ``scaler`` switches on mixed precision (GPU only, about 2x faster)."""
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss, seen = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(images).squeeze(1).float(), labels)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    return total_loss / seen


@torch.no_grad()
def predict(model, loader: DataLoader, device, criterion: nn.Module | None = None):
    """Return (P(fake) per image, true labels, mean loss or nan). Uses mixed precision on a GPU."""
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True)).squeeze(1).float()
        if criterion is not None:
            total_loss += criterion(logits, labels.float().to(device)).item() * len(labels)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    loss = total_loss / len(labels) if criterion is not None else float("nan")
    return probs, labels.astype(np.int64), loss
