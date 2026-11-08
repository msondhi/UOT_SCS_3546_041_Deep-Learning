"""Real-world image degradations for robustness testing and augmentation.

Each degradation takes a severity from 1 (mild) to 5 (severe) and works on PIL
images, so it can be dropped into a torchvision ``transforms.Compose``.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter


class Degradation:
    """Base class: subclasses define ``levels`` (one parameter per severity) and ``apply``."""

    levels: tuple = ()

    def __init__(self, severity: int = 3):
        if not 1 <= severity <= len(self.levels):
            raise ValueError(f"severity must be 1-{len(self.levels)}, got {severity}")
        self.severity = severity
        self.param = self.levels[severity - 1]

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.apply(img.convert("RGB"))

    def apply(self, img: Image.Image) -> Image.Image:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(severity={self.severity})"


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class JPEGCompression(Degradation):
    """Lossy compression, like images sent through messaging apps. Param: JPEG quality."""

    levels = (90, 70, 50, 30, 10)

    def apply(self, img):
        return _jpeg(img, self.param)


class GaussianBlur(Degradation):
    """Out-of-focus or low-quality camera. Param: blur radius in pixels."""

    levels = (0.5, 1.0, 1.5, 2.5, 4.0)

    def apply(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.param))


class Downscale(Degradation):
    """Low-resolution upload, shrunk then stretched back. Param: scale factor."""

    levels = (0.75, 0.5, 0.35, 0.25, 0.15)

    def apply(self, img):
        w, h = img.size
        small = img.resize((max(1, int(w * self.param)), max(1, int(h * self.param))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class GaussianNoise(Degradation):
    """Sensor noise from poor lighting. Param: noise std-dev in pixel values (0-255)."""

    levels = (5, 10, 20, 30, 45)

    def apply(self, img):
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.default_rng().normal(0, self.param, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class ScreenCapture(Degradation):
    """Photo of a screen: moire stripes, blue colour cast, slight blur and compression.

    Param: strength of the moire pattern.
    """

    levels = (0.03, 0.06, 0.09, 0.12, 0.15)

    def apply(self, img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        angle, period = 0.3, 4.0
        stripes = np.sin(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) / period)
        arr = arr * (1 + self.param * stripes[..., None])
        arr = (arr * np.array([1.0, 1.02, 1.06])) ** 1.1
        out = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        return _jpeg(out.filter(ImageFilter.GaussianBlur(0.8)), 75)


DEGRADATIONS: dict[str, type[Degradation]] = {
    "jpeg": JPEGCompression,
    "blur": GaussianBlur,
    "downscale": Downscale,
    "noise": GaussianNoise,
    "screen": ScreenCapture,
}


class RandomDegradation:
    """Training augmentation: with probability ``p``, apply one random degradation at a random severity."""

    def __init__(self, p: float = 0.5, max_severity: int = 5):
        self.p = p
        self.max_severity = max_severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        cls = random.choice(list(DEGRADATIONS.values()))
        return cls(random.randint(1, self.max_severity))(img)

    def __repr__(self) -> str:
        return f"RandomDegradation(p={self.p}, max_severity={self.max_severity})"
