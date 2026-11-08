"""End-to-end: train -> evaluate (with degradations) -> Grad-CAM, on the dummy dataset."""

from conftest import REPO_ROOT

from deepfake_kyc import evaluate, explain, train


def test_train_evaluate_explain(tiny_overrides, tmp_path):
    config = str(REPO_ROOT / "configs/experiments/01_baseline_cnn.yaml")
    summary = train.main(["--config", config, "--set", *tiny_overrides])
    ckpt = tmp_path / "01_baseline_cnn" / "best.pt"
    assert ckpt.exists() and summary["best_epoch"] == 1
    assert (tmp_path / "01_baseline_cnn" / "history.csv").exists()

    rows = evaluate.main(["--checkpoint", str(ckpt), "--degradation", "jpeg", "blur", "--severities", "1", "5"])
    assert [(r["condition"], r["severity"]) for r in rows] == [
        ("clean", 0), ("jpeg", 1), ("jpeg", 5), ("blur", 1), ("blur", 5)
    ]
    assert (tmp_path / "01_baseline_cnn" / "eval_test.csv").exists()

    png = explain.main(["--checkpoint", str(ckpt), "--num-images", "2"])
    assert png.exists()
