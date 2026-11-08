"""Model definitions and helpers.

Every model outputs a single logit per image; sigmoid(logit) = P(fake).
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small baseline trained from scratch: four conv blocks, global pooling, linear head."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(name: str, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """``simple_cnn`` or any timm model name (e.g. ``efficientnet_b0``, ``resnet50``, ``vit_small_patch16_224``)."""
    if name == "simple_cnn":
        return SimpleCNN(dropout)
    return timm.create_model(name, pretrained=pretrained, num_classes=1, drop_rate=dropout)


def head_module(model: nn.Module) -> nn.Module:
    return model.head if isinstance(model, SimpleCNN) else model.get_classifier()


def split_parameters(model: nn.Module) -> tuple[list, list]:
    """Return (backbone_params, head_params) so they can get different learning rates."""
    head_ids = {id(p) for p in head_module(model).parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return backbone, head


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    backbone, _ = split_parameters(model)
    for p in backbone:
        p.requires_grad = trainable


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage, which Grad-CAM uses to build its heatmap."""
    if isinstance(model, SimpleCNN):
        return model.features[-1]
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4[-1]
    if hasattr(model, "bn2"):  # EfficientNet
        return model.bn2
    raise ValueError(f"No Grad-CAM target layer defined for {type(model).__name__}")


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg["model"]["name"], pretrained=False, dropout=cfg["model"]["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg
