import numpy as np
import pytest

from highspl.dsp import beamformer as bf

SR = 48_000


def _noise(level, seed, n=512):
    return (level * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def test_blend_uses_primary_only_when_quiet_and_beamformer_when_loud():
    fe = bf.BlendedFrontEnd(
        bf.DifferentialFrontEnd(0.015, 0.055, SR), lo_dbfs=-34.0, hi_dbfs=-24.0
    )

    for i in range(80):  # quiet: about -60 dBFS
        fe.process(_noise(1e-3, i), _noise(1e-3, 1000 + i))
    assert fe.alpha == pytest.approx(0.0, abs=1e-6)

    fe.reset()
    for i in range(80):  # loud: about -14 dBFS
        fe.process(_noise(0.2, i), _noise(0.2, 1000 + i))
    assert fe.alpha == pytest.approx(1.0, abs=1e-6)


def test_blend_quiet_output_is_the_delay_aligned_primary():
    inner = bf.DifferentialFrontEnd(0.015, 0.055, SR)
    fe = bf.BlendedFrontEnd(inner, lo_dbfs=-34.0, hi_dbfs=-24.0)
    fe.reset()
    fe.alpha = 0.0
    x = _noise(1e-3, 1, 512 * 40)
    r = _noise(1e-3, 2, 512 * 40)
    out = np.concatenate(
        [fe.process(x[i : i + 512], r[i : i + 512]) for i in range(0, x.size, 512)]
    )

    d = inner.delay_samples
    # alpha first slews up (noise level unknown for 8 blocks) and back down; compare after it settled
    k = 512 * 30
    np.testing.assert_allclose(out[k:], x[k - d : out.size - d], atol=1e-6)


def test_blend_alpha_is_slew_limited():
    fe = bf.BlendedFrontEnd(
        bf.DifferentialFrontEnd(0.015, 0.055, SR), slew_per_block=0.1
    )
    fe.alpha = 1.0
    quiet = np.full(512, 1e-4, dtype=np.float32)
    for _ in range(20):
        fe._tracker.update(quiet)
    before = fe.alpha
    fe.process(quiet, quiet)

    assert before - fe.alpha <= 0.1 + 1e-9


def test_blend_output_is_finite_and_continuous():
    fe = bf.BlendedFrontEnd(bf.DifferentialFrontEnd(0.015, 0.055, SR))
    out = np.concatenate(
        [fe.process(_noise(0.1, i), _noise(0.1, 500 + i)) for i in range(60)]
    )

    assert np.isfinite(out).all()
    assert np.abs(np.diff(out)).max() < 8 * np.abs(np.diff(out)).std()
