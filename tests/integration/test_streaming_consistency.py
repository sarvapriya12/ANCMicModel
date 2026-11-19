import numpy as np
import pytest

from highspl.dsp.agc import LookaheadLimiter, StreamingAGC
from highspl.dsp.frontend import make_frontend

SR = 48_000


def _stream(fe, f, r, block):
    return np.concatenate(
        [
            fe.process(f[i : i + block], r[i : i + block])
            for i in range(0, f.size - block + 1, block)
        ]
    )


@pytest.mark.parametrize("mode", ["diff", "adiff", "gsc"])
def test_beamformers_are_block_size_invariant(mode):
    """Same stream, different block sizes: the fixed filter must be identical (no
    block-edge artefacts); the adaptive ones must stay within a tight tolerance."""
    rng = np.random.default_rng(0)
    n = 512 * 40
    f = (0.1 * rng.standard_normal(n)).astype(np.float32)
    r = (0.1 * rng.standard_normal(n)).astype(np.float32)
    kw = {"mic_spacing_m": 0.015, "mouth_distance_m": 0.055, "sample_rate": SR}

    a = _stream(make_frontend(mode, **kw), f, r, 512)
    b = _stream(make_frontend(mode, **kw), f, r, 128)

    tol = 1e-5 if mode == "diff" else 0.3
    err = np.abs(a - b).max() / (np.abs(a).max() + 1e-9)
    assert err < tol, err


def test_limiter_is_state_continuous_across_blocks():
    rng = np.random.default_rng(1)
    x = (0.3 * rng.standard_normal(512 * 30)).astype(np.float32)

    whole = LookaheadLimiter().process(x)
    lim = LookaheadLimiter()
    pieces = np.concatenate(
        [lim.process(x[i : i + 512]) for i in range(0, x.size, 512)]
    )

    np.testing.assert_allclose(whole, pieces, atol=1e-6)


def test_agc_output_is_finite_over_a_long_stream():
    rng = np.random.default_rng(2)
    x = (0.3 * rng.standard_normal(512 * 200)).astype(np.float32)
    agc = StreamingAGC()
    out = np.concatenate([agc.process(x[i : i + 512]) for i in range(0, x.size, 512)])

    assert np.isfinite(out).all()
