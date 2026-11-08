"""Small shared helpers: seeding, device selection, JSON output."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Prefer an NVIDIA GPU (Colab), then Apple Silicon GPU (MPS), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
