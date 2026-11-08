from conftest import REPO_ROOT

from deepfake_kyc.config import apply_overrides, load_config


def test_experiment_inherits_base_and_overrides_it():
    cfg = load_config(REPO_ROOT / "configs/experiments/01_baseline_cnn.yaml")
    assert cfg["model"]["name"] == "simple_cnn"
    assert cfg["data"]["image_size"] == 224  # inherited from base.yaml
    assert "base" not in cfg


def test_overrides_parse_yaml_types():
    cfg = apply_overrides({"data": {"batch_size": 64}}, ["data.batch_size=8", "data.train_subset=null", "x.y=true"])
    assert cfg == {"data": {"batch_size": 8, "train_subset": None}, "x": {"y": True}}


def test_every_experiment_config_loads():
    for path in sorted((REPO_ROOT / "configs/experiments").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg["experiment_name"] == path.stem
