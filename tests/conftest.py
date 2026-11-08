from pathlib import Path

import pytest

from deepfake_kyc.dummy_data import make_dummy_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dummy_root(tmp_path_factory) -> Path:
    return make_dummy_dataset(tmp_path_factory.mktemp("data"), per_class=8, size=32)


@pytest.fixture
def tiny_overrides(dummy_root, tmp_path) -> list[str]:
    """Config overrides for a fast CPU-sized run on the dummy dataset."""
    return [
        f"data.root={dummy_root}",
        "data.image_size=32",
        "data.batch_size=4",
        "data.num_workers=0",
        f"output_dir={tmp_path}",
        "train.epochs=1",
    ]
