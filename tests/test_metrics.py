import pytest

from deepfake_kyc.metrics import compute_metrics


def test_far_and_frr():
    # labels: 1 = fake, 0 = real
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.1, 0.1, 0.2, 0.3, 0.7]
    m = compute_metrics(labels, probs)
    assert m["far"] == pytest.approx(0.5)  # 2 of 4 fakes accepted as real
    assert m["frr"] == pytest.approx(0.25)  # 1 of 4 real customers rejected
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 2, 1, 3)


def test_perfect_predictions():
    m = compute_metrics([0, 1], [0.1, 0.9])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and m["far"] == 0.0 and m["frr"] == 0.0
