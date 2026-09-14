import numpy as np
import pytest

from highspl.dsp.gain import Gain


def test_unity_gain():
    gain = Gain(db=0.0)

    audio = np.array([0.0, 0.25, -0.5, 1.0], dtype=np.float32)

    output = gain.process(audio)

    assert output.dtype == np.float32
    assert np.array_equal(output, audio)
    assert output is not audio


def test_positive_gain():
    gain = Gain(db=6.0)

    audio = np.array([0.25, -0.5, 1.0], dtype=np.float32)

    output = gain.process(audio)

    expected = audio * (10.0 ** (6.0 / 20.0))

    assert np.allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_negative_gain():
    gain = Gain(db=-6.0)

    audio = np.array([0.25, -0.5, 1.0], dtype=np.float32)

    output = gain.process(audio)

    expected = audio * (10.0 ** (-6.0 / 20.0))

    assert np.allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_gain_preserves_shape():
    gain = Gain(db=3.0)

    audio = np.ones((100, 2), dtype=np.float32)

    output = gain.process(audio)

    assert output.shape == audio.shape
    assert output.dtype == np.float32


def test_non_finite_input_is_rejected():
    gain = Gain(db=0.0)

    audio = np.zeros(10, dtype=np.float32)
    audio[5] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        gain.process(audio)


@pytest.mark.parametrize("db", [np.nan, np.inf, -np.inf])
def test_invalid_gain_is_rejected(db):
    with pytest.raises(ValueError, match="gain"):
        Gain(db=db)


def test_invalid_input_type_is_rejected():
    gain = Gain(db=0.0)

    with pytest.raises(TypeError, match="array"):
        gain.process([1.0, 2.0, 3.0])
