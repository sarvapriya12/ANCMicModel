import numpy as np
import pytest

from highspl.dsp.limiter import PeakLimiter


def test_signal_below_ceiling_is_unchanged():
    limiter = PeakLimiter(ceiling_db=-1.0)

    audio = np.array(
        [0.0, 0.1, -0.25, 0.5],
        dtype=np.float32,
    )

    output = limiter.process(audio)

    assert output.dtype == np.float32
    assert np.allclose(output, audio)
    assert output is not audio


def test_peak_is_limited_to_ceiling():
    limiter = PeakLimiter(ceiling_db=-1.0)

    audio = np.array(
        [0.0, 0.5, -1.0, 2.0, -3.0],
        dtype=np.float32,
    )

    output = limiter.process(audio)

    ceiling = 10.0 ** (-1.0 / 20.0)

    assert np.max(np.abs(output)) <= ceiling + 1e-6


def test_positive_and_negative_peaks_are_limited():
    limiter = PeakLimiter(ceiling_db=-6.0)

    audio = np.array(
        [2.0, -3.0],
        dtype=np.float32,
    )

    output = limiter.process(audio)

    ceiling = 10.0 ** (-6.0 / 20.0)

    assert np.max(np.abs(output)) <= ceiling + 1e-6
    assert output[0] > 0
    assert output[1] < 0


def test_limiter_preserves_shape():
    limiter = PeakLimiter(ceiling_db=-1.0)

    audio = np.ones((100, 2), dtype=np.float32)

    output = limiter.process(audio)

    assert output.shape == audio.shape
    assert output.dtype == np.float32


def test_limiter_does_not_clip_individual_samples():
    limiter = PeakLimiter(ceiling_db=-6.0)

    audio = np.array(
        [0.5, 1.0, 2.0],
        dtype=np.float32,
    )

    output = limiter.process(audio)

    # A peak limiter should scale the block rather than hard-clip
    # individual samples independently.
    ratio = output[2] / audio[2]

    assert np.allclose(
        output,
        audio * ratio,
        rtol=1e-5,
        atol=1e-5,
    )


def test_non_finite_input_is_rejected():
    limiter = PeakLimiter(ceiling_db=-1.0)

    audio = np.zeros(10, dtype=np.float32)
    audio[5] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        limiter.process(audio)


@pytest.mark.parametrize(
    "ceiling_db",
    [np.nan, np.inf, -np.inf],
)
def test_invalid_ceiling_is_rejected(ceiling_db):
    with pytest.raises(ValueError, match="ceiling"):
        PeakLimiter(ceiling_db=ceiling_db)


def test_invalid_input_type_is_rejected():
    limiter = PeakLimiter(ceiling_db=-1.0)

    with pytest.raises(TypeError, match="array"):
        limiter.process([1.0, 2.0, 3.0])
