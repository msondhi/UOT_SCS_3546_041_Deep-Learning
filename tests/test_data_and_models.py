from pathlib import Path

import pytest
import torch
from conftest import REPO_ROOT
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from deepfake_kyc.config import load_config
from deepfake_kyc.data import build_loader, make_dataset
from deepfake_kyc.engine import train_one_epoch
from deepfake_kyc.models import build_model, gradcam_target_layer, set_backbone_trainable, split_parameters


def test_fake_is_labelled_one(dummy_root):
    ds = make_dataset(dummy_root / "train")
    for i, (path, _) in enumerate(ds.samples):
        _, label = ds[i]
        assert label == (1 if Path(path).parent.name == "fake" else 0)


def test_loader_shapes_and_subset(dummy_root, tiny_overrides):
    cfg = load_config(REPO_ROOT / "configs/base.yaml", tiny_overrides + ["data.train_subset=6"])
    loader = build_loader(cfg, "train", train=True)
    assert len(loader.dataset) == 6
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 32, 32) and labels.dtype == torch.int64


@pytest.mark.parametrize("name", ["simple_cnn", "efficientnet_b0", "resnet50"])
def test_models_output_one_logit_and_freeze(name):
    model = build_model(name, pretrained=False)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 1)
    gradcam_target_layer(model)

    set_backbone_trainable(model, False)
    backbone, head = split_parameters(model)
    assert backbone and head
    assert not any(p.requires_grad for p in backbone)
    assert all(p.requires_grad for p in head)


def test_frozen_batch_norm_statistics_do_not_update():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))
    model[0].weight.requires_grad_(False)
    model[0].bias.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )
    initial_mean = model[0].running_mean.clone()
    initial_var = model[0].running_var.clone()
    initial_batches = model[0].num_batches_tracked.clone()
    initial_head = model[1].weight.detach().clone()

    train_one_epoch(
        model, loader, nn.BCEWithLogitsLoss(), optimizer, scheduler, torch.device("cpu"), freeze_batch_norm=True
    )

    assert torch.equal(model[0].running_mean, initial_mean)
    assert torch.equal(model[0].running_var, initial_var)
    assert torch.equal(model[0].num_batches_tracked, initial_batches)
    assert not torch.equal(model[1].weight, initial_head)
