import numpy as np
import pytest

from highspl.dsp.agc import LookaheadLimiter, StreamingAGC

SR = 48_000


def _blocks(x, n=512):
    return [x[i : i + n] for i in range(0, x.size - n + 1, n)]


def _stream(proc, x):
    return np.concatenate([proc.process(b) for b in _blocks(x)])


def test_limiter_never_exceeds_ceiling_even_across_blocks():
    rng = np.random.default_rng(0)
    x = 0.2 * rng.standard_normal(SR).astype(np.float32)
    for pos in (3, 511, 512, 1000, 20000):  # peaks on and across block edges
        x[pos] = 8.0
    lim = LookaheadLimiter(ceiling_dbfs=-1.0)
    y = _stream(lim, x)

    assert np.abs(y).max() <= 10 ** (-1.0 / 20) + 1e-4


def test_limiter_is_transparent_below_the_ceiling():
    rng = np.random.default_rng(1)
    x = 0.1 * rng.standard_normal(SR).astype(np.float32)
    lim = LookaheadLimiter(ceiling_dbfs=-1.0)
    y = _stream(lim, x)

    d = lim.latency
    np.testing.assert_allclose(y[2000:], x[2000 - d : y.size - d], atol=1e-5)


def test_limiter_gain_ramps_before_the_peak_not_after():
    x = np.full(SR // 2, 0.3, dtype=np.float32)
    x[10000] = 6.0
    lim = LookaheadLimiter(ceiling_dbfs=-1.0, lookahead=48, hold=480)
    y = _stream(lim, x)
    d = lim.latency
    g = y[: x.size - d + 0] / np.maximum(x[: y.size][0:0].sum() + 1, 1)  # unused
    del g
    gain = y[d : d + 20000] / x[:20000]

    peak_out = 10000
    assert gain[peak_out - 60] > 0.99  # untouched well before
    assert gain[peak_out - 10] < 0.9  # already ducking ahead of the peak
    assert gain[peak_out + 100] < 0.5  # held down after
    assert gain[peak_out + 2000] > 0.99  # released


def test_limiter_latency_is_lookahead():
    assert LookaheadLimiter(lookahead=48).latency == 48


def test_agc_raises_quiet_and_lowers_loud_towards_target():
    rng = np.random.default_rng(2)
    seconds = 12
    quiet = (0.003 * rng.standard_normal(SR * seconds)).astype(np.float32)  # ~ -50 dBFS
    loud = (0.7 * rng.standard_normal(SR * seconds)).astype(np.float32)  # ~ -3 dBFS

    def settled_db(x):
        agc = StreamingAGC(target_dbfs=-24.0, max_gain_db=30.0)
        y = _stream(agc, x)
        tail = y[-SR:]
        return 10 * np.log10(np.mean(tail**2))

    assert settled_db(quiet) == pytest.approx(-24.0, abs=3.0)
    assert settled_db(loud) == pytest.approx(-24.0, abs=3.0)


def test_agc_gain_is_bounded():
    agc = StreamingAGC(target_dbfs=-24.0, max_gain_db=12.0, min_gain_db=-6.0)
    quiet = (1e-4 * np.random.default_rng(3).standard_normal(SR * 6)).astype(np.float32)
    _stream(agc, quiet)

    assert agc.gain_db <= 12.0 + 1e-6

    agc = StreamingAGC(target_dbfs=-24.0, max_gain_db=12.0, min_gain_db=-6.0)
    loud = (0.9 * np.random.default_rng(4).standard_normal(SR * 6)).astype(np.float32)
    _stream(agc, loud)
    assert agc.gain_db >= -6.0 - 1e-6


def test_agc_holds_gain_through_an_impulse_instead_of_pumping():
    rng = np.random.default_rng(5)
    x = (0.05 * rng.standard_normal(SR * 6)).astype(np.float32)
    agc = StreamingAGC(target_dbfs=-24.0)
    _stream(agc, x[: SR * 3])
    before = agc.gain_db

    bang = x[SR * 3 : SR * 3 + 512].copy()
    bang[100] = 1.0
    agc.process(bang)

    assert agc.gain_db == pytest.approx(before, abs=1e-9)


def test_agc_digital_silence_does_not_blow_up():
    agc = StreamingAGC(max_gain_db=24.0)
    y = _stream(agc, np.zeros(SR * 3, dtype=np.float32))

    assert np.isfinite(y).all() and np.abs(y).max() == 0.0


def test_agc_and_limiter_handle_empty_and_short_blocks():
    agc, lim = StreamingAGC(), LookaheadLimiter()
    assert agc.process(np.empty(0, dtype=np.float32)).size == 0
    out = lim.process(np.zeros(7, dtype=np.float32))
    assert out.shape == (7,)
