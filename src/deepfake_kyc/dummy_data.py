"""Tiny synthetic real/fake dataset for code checks and tests (not for real experiments).

"Real" images are smooth colour gradients and "fake" images are noisy checkerboards,
so even a small model can learn the difference in one epoch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_dummy_dataset(out: str | Path, per_class: int = 16, size: int = 64, seed: int = 0) -> Path:
    out = Path(out)
    rng = np.random.default_rng(seed)
    for split in ("train", "valid", "test"):
        for label in ("real", "fake"):
            folder = out / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                if label == "real":
                    ramp = np.linspace(0, 1, size)[None, :, None]
                    colour = rng.uniform(0.3, 1.0, size=(1, 1, 3))
                    arr = np.broadcast_to(ramp * colour, (size, size, 3))
                else:
                    tile = (np.indices((size, size)).sum(axis=0) // 8) % 2
                    arr = 0.5 * tile[..., None] + rng.uniform(0, 0.5, size=(size, size, 3))
                Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(folder / f"{i:03d}.jpg")
    return out
