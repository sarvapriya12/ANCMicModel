import numpy as np
import pytest

from highspl.data import rir
from highspl.dsp.calibration import RearCalibrator, estimate_mouth_path
from highspl.dsp.drift import DriftCompensator

SR = 48_000


def _speechlike(n, seed):
    x = np.random.default_rng(seed).standard_normal(n)
    return np.convolve(x, np.ones(4) / 4, mode="same")


# ------------------------------------------------------------ calibration


@pytest.mark.parametrize("gain,delay", [(0.79, 2.1), (0.7, 2.8), (0.9, 1.6)])
def test_mouth_path_estimator_recovers_gain_and_delay(gain, delay):
    f = _speechlike(SR * 3, 1)
    r = gain * rir.fractional_delay(f, delay)

    est = estimate_mouth_path(f, r, SR)

    assert est["gain"] == pytest.approx(gain, rel=0.03)
    assert est["delay"] == pytest.approx(delay, abs=0.1)


def test_estimator_is_robust_to_moderate_noise():
    f = _speechlike(SR * 4, 2)
    rng = np.random.default_rng(3)
    r = 0.79 * rir.fractional_delay(f, 2.1) + 0.2 * f.std() * rng.standard_normal(
        f.size
    )

    est = estimate_mouth_path(f + 0.1 * f.std() * rng.standard_normal(f.size), r, SR)

    assert est["gain"] == pytest.approx(0.79, rel=0.1)
    assert est["delay"] == pytest.approx(2.1, abs=0.3)


def test_estimator_reports_nan_when_signals_are_unrelated():
    rng = np.random.default_rng(4)
    est = estimate_mouth_path(rng.standard_normal(SR), rng.standard_normal(SR), SR)

    assert np.isnan(est["gain"])


def test_rear_calibrator_restores_nominal_pod_response():
    g_nom, tau_nom = 0.786, 2.1
    f = _speechlike(SR * 3, 5)
    # device with +0.8 dB rear gain error and a 0.7-sample I2S offset
    r = 1.096 * g_nom * rir.fractional_delay(f, tau_nom + 0.7)

    cal = RearCalibrator(g_nom, tau_nom)
    cal.set_from_measurement(
        **{
            k: v
            for k, v in zip(
                ("gain_eff", "delay_eff"),
                (
                    estimate_mouth_path(f, r, SR)["gain"],
                    estimate_mouth_path(f, r, SR)["delay"],
                ),
            )
        }
    )
    out_f, out_r = [], []
    for i in range(0, f.size - 511, 512):
        a, b = cal.process(
            f[i : i + 512].astype(np.float32), r[i : i + 512].astype(np.float32)
        )
        out_f.append(a), out_r.append(b)
    f2, r2 = np.concatenate(out_f), np.concatenate(out_r)

    # after correction r2 ~ g_nom * f2 delayed by tau_nom
    nominal = g_nom * rir.fractional_delay(f2.astype(np.float64), tau_nom)
    err = r2[3000:] - nominal[3000:]
    assert 10 * np.log10(np.mean(err**2) / np.mean(nominal[3000:] ** 2)) < -25.0


def test_disabled_calibrator_is_identity_with_no_delay():
    cal = RearCalibrator(0.8, 2.0)
    f = np.arange(512, dtype=np.float32)
    a, _b = cal.process(f, f * 2)

    np.testing.assert_array_equal(a, f)
    assert cal.latency == 0


def test_calibrator_refuses_nonsense_measurements():
    cal = RearCalibrator(0.8, 2.0)
    cal.set_from_measurement(float("nan"), 1.0)
    assert not cal.enabled
    cal.set_from_measurement(-1.0, 1.0)
    assert not cal.enabled


# ------------------------------------------------------------------ drift


def _simulate(ppm_producer, blocks, comp=True, block=512):
    """Producer runs ``ppm`` fast relative to the consumer (the callback)."""
    fifo = DriftCompensator(target_fill=2 * block)
    t_prod, t_cons = 0.0, 0.0
    period_p = block / (SR * (1 + ppm_producer * 1e-6))
    period_c = block / SR
    raw_fill = 0.0
    fills = []
    x = np.sin(2 * np.pi * 440 * np.arange(block) / SR).astype(np.float32)
    # prime the FIFO to the target
    fifo.write(np.zeros(2 * block, dtype=np.float32))
    raw_fill = 2 * block
    for _ in range(blocks):
        t_prod += period_p
        if comp:
            fifo.write(x)
        raw_fill += block
        while t_cons + period_c <= t_prod:
            t_cons += period_c
            if comp:
                fifo.read(block)
            raw_fill -= block
        fills.append(fifo.fill if comp else raw_fill)
    return fifo, np.array(fills)


def test_without_compensation_50ppm_drifts_without_bound():
    _, fills = _simulate(50.0, 60_000, comp=False)  # ~10 minutes

    assert abs(fills[-1] - fills[0]) > 512  # a whole block lost/invented


def test_pi_controller_holds_fifo_fill_and_avoids_xruns_at_50ppm():
    fifo, fills = _simulate(50.0, 60_000, comp=True)

    assert fifo.underruns == 0 and fifo.overruns == 0
    # within one block (the consumer drains in whole blocks) of the target
    assert np.abs(fills[3000:] - 1024).max() <= 600
    assert abs(fifo.ppm) < 500  # tiny ratio change, inaudible


def test_controller_tracks_negative_drift_too():
    fifo, fills = _simulate(-80.0, 60_000, comp=True)

    assert fifo.underruns == 0 and fifo.overruns == 0
    assert np.abs(fills[3000:] - 1024).max() <= 600


def test_resampler_preserves_a_tone():
    fifo = DriftCompensator(target_fill=1024)
    fifo.write(np.zeros(1024, dtype=np.float32))
    t = np.arange(48_000) / SR
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    outs = []
    for i in range(0, x.size, 512):
        fifo.write(x[i : i + 512])
        outs.append(fifo.read(512))
    y = np.concatenate(outs)[4096:]
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    peak = np.fft.rfftfreq(y.size, 1 / SR)[np.argmax(spec)]

    assert peak == pytest.approx(1000.0, abs=15.0)


def test_read_underrun_is_zero_filled_and_counted():
    fifo = DriftCompensator(target_fill=512)
    fifo.write(np.ones(100, dtype=np.float32))
    out = fifo.read(512)

    assert out.shape == (512,) and fifo.underruns == 1
    assert (
        out[1:90].min() > 0.0 and out[-1] == 0.0
    )  # out[0] is the filter's start-up sample
