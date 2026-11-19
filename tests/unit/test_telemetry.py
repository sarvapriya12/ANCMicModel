import numpy as np
import pytest

from highspl.telemetry import NoiseFloorTracker, latency_budget


def _feed(tracker, level, n=40, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        tracker.update(level * rng.standard_normal(512))


def test_floor_is_nan_until_enough_blocks():
    t = NoiseFloorTracker()
    _feed(t, 0.01, n=3)

    assert np.isnan(t.floor_dbfs)


def test_floor_tracks_the_quiet_pauses_not_the_speech():
    t = NoiseFloorTracker(seconds=4.0)
    rng = np.random.default_rng(1)
    for i in range(300):
        loud = (i // 20) % 2 == 0  # speech bursts alternate with pauses
        t.update((0.2 if loud else 0.002) * rng.standard_normal(512))

    assert t.floor_dbfs == pytest.approx(20 * np.log10(0.002), abs=3.0)
    assert t.speech_level_dbfs > t.floor_dbfs + 30


def test_nonfinite_samples_do_not_poison_the_floor():
    t = NoiseFloorTracker()
    _feed(t, 0.01, n=20)
    bad = np.full(512, 0.01)
    bad[3] = np.nan
    bad[9] = np.inf
    t.update(bad)
    _feed(t, 0.01, n=20, seed=2)

    assert np.isfinite(t.floor_dbfs)


def test_muting_everything_is_not_reported_as_a_noise_reduction_of_speech():
    """The old metric (10*log10(E_out/E_in)) scored a muted output as perfect.
    The floor of a muted stream is far below the input floor, but its *speech
    level* collapses too, which is what makes the two distinguishable."""
    inp, muted = NoiseFloorTracker(), NoiseFloorTracker()
    rng = np.random.default_rng(3)
    for i in range(300):
        x = (0.2 if (i // 20) % 2 == 0 else 0.002) * rng.standard_normal(512)
        inp.update(x)
        muted.update(1e-6 * x)

    assert muted.speech_level_dbfs < inp.speech_level_dbfs - 60


def test_latency_budget_is_the_sum_of_its_parts():
    b = latency_budget(
        512, model_latency_samples=512, limiter_samples=48, compute_ms=2.0
    )

    assert b["capture_block_ms"] == pytest.approx(10.667, abs=1e-3)
    assert b["model_ms"] == pytest.approx(10.667, abs=1e-3)
    assert b["total_lower_bound_ms"] == pytest.approx(
        b["capture_block_ms"]
        + b["model_ms"]
        + b["limiter_ms"]
        + 2.0
        + b["playback_block_ms"]
    )
    assert 32.0 < b["total_lower_bound_ms"] < 36.0
