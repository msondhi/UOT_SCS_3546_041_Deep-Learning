import numpy as np
import pytest
from PIL import Image

from deepfake_kyc.degradations import DEGRADATIONS, RandomDegradation


@pytest.fixture
def face():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
@pytest.mark.parametrize("severity", [1, 5])
def test_degradation_keeps_size_and_changes_pixels(face, name, severity):
    out = DEGRADATIONS[name](severity)(face)
    assert out.size == face.size and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(face))


def test_higher_severity_blurs_more(face):
    def diff(severity):
        out = DEGRADATIONS["blur"](severity)(face)
        return np.abs(np.asarray(out, float) - np.asarray(face, float)).mean()

    assert diff(5) > diff(1)


def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        DEGRADATIONS["jpeg"](6)


def test_random_degradation_with_p_zero_is_identity(face):
    assert RandomDegradation(p=0.0)(face) is face
