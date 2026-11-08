"""Datasets, transforms and data loaders.

Expected folder layout (as unzipped from Kaggle)::

    <data.root>/train/{real,fake}/*.jpg
    <data.root>/valid/{real,fake}/*.jpg
    <data.root>/test/{real,fake}/*.jpg

Labels: fake = 1 (the class we want to catch), real = 0.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .degradations import Degradation, RandomDegradation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


class FakeIsPositive:
    """Remap ImageFolder's alphabetical class index so that fake = 1 and real = 0."""

    def __init__(self, fake_idx: int):
        self.fake_idx = fake_idx

    def __call__(self, target: int) -> int:
        return int(target == self.fake_idx)


def build_transform(
    image_size: int,
    train: bool = False,
    standard: bool = True,
    degradation: bool = False,
    degradation_prob: float = 0.5,
    eval_degradation: Degradation | None = None,
) -> transforms.Compose:
    steps = []
    if eval_degradation is not None:
        steps.append(eval_degradation)
    if train and degradation:
        steps.append(RandomDegradation(p=degradation_prob))
    if train and standard:
        steps += [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    else:
        steps.append(transforms.Resize((image_size, image_size)))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def make_dataset(root: str | Path, transform=None) -> datasets.ImageFolder:
    ds = datasets.ImageFolder(str(root), transform=transform)
    if set(ds.class_to_idx) != {"fake", "real"}:
        raise ValueError(f"{root} must contain exactly two folders, 'fake' and 'real'; found {ds.classes}")
    ds.target_transform = FakeIsPositive(ds.class_to_idx["fake"])
    return ds


def subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    """Random subset of ``n`` items (reproducible); returns ``ds`` unchanged if n is None or too large."""
    if not n or n >= len(ds):
        return ds
    generator = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=generator)[:n].tolist())


def build_loader(
    cfg: dict,
    split: str,
    train: bool = False,
    eval_degradation: Degradation | None = None,
    root_override: str | Path | None = None,
) -> DataLoader:
    """Loader for ``split`` (train/val/test). ``root_override`` points at another real/fake folder (e.g. OOD set)."""
    d, aug = cfg["data"], cfg["augment"]
    root = Path(root_override) if root_override else Path(d["root"]) / SPLIT_DIRS[split]
    transform = build_transform(
        d["image_size"],
        train=train,
        standard=aug["standard"],
        degradation=aug["degradation"],
        degradation_prob=aug["degradation_prob"],
        eval_degradation=eval_degradation,
    )
    ds = subsample(make_dataset(root, transform), d.get(f"{split}_subset"), cfg["seed"])
    return DataLoader(
        ds,
        batch_size=d["batch_size"],
        shuffle=train,
        num_workers=d["num_workers"],
        persistent_workers=d["num_workers"] > 0,
    )
