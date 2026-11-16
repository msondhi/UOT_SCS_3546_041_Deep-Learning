from pathlib import Path

import pytest
import torch
from conftest import REPO_ROOT

from deepfake_kyc.config import load_config
from deepfake_kyc.data import build_loader, make_dataset
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
