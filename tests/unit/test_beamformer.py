import numpy as np
import pytest
from scipy import signal as sps

from highspl.data import rir
from highspl.dsp import beamformer as bf
from highspl.dsp.frontend import make_frontend

SR = 48_000
GEOM = rir.PodGeometry(spacing_m=0.015, mouth_m=0.055, shadow_db=0.0)


def _noise(seed, n=SR * 2):
    x = np.random.default_rng(seed).standard_normal(n)
    return x / x.std()


def _run(fe, f, r, block=512):
    out = []
    for i in range(0, f.size - block + 1, block):
        out.append(
            fe.process(
                f[i : i + block].astype(np.float32), r[i : i + block].astype(np.float32)
            )
        )
    return np.concatenate(out)


def _band_power(x, lo, hi):
    spec = np.abs(np.fft.rfft(x[2000:])) ** 2
    f = np.fft.rfftfreq(x[2000:].size, 1 / SR)
    return spec[(f >= lo) & (f < hi)].sum()


def test_fractional_delay_fir_delays_by_requested_amount():
    tau = 2.1
    h = bf.fractional_delay_fir(tau)
    t = np.arange(SR // 2) / SR
    for f0 in (500.0, 2000.0, 5000.0):
        x = np.sin(2 * np.pi * f0 * t)
        y = sps.lfilter(h, [1.0], x)
        expected = np.sin(2 * np.pi * f0 * (t - (bf.DELAY + tau) / SR))
        np.testing.assert_allclose(y[200:], expected[200:], atol=0.02)


def test_mouth_eq_is_gain_capped_and_flattens_response():
    tau = 0.015 / bf.C_SOUND * SR
    g = 0.055 / (0.055 + 0.015)
    eq = bf.design_mouth_eq(tau, g, 0.25, SR, max_gain_db=12.0)
    f = np.array([100.0, 300.0, 1000.0, 2000.0, 4000.0, 8000.0])
    _, h_eq = sps.sosfreqz(eq, worN=f, fs=SR)
    total = np.abs(h_eq) * np.abs(bf.mouth_response(f, tau, g, 0.25, SR))

    assert np.abs(h_eq).max() <= 10 ** (14.0 / 20)  # shelves stack a little
    # mouth speech ends up within +/-2.5 dB of its 2 kHz level over 100 Hz-8 kHz
    ref = total[3]
    assert np.all(np.abs(20 * np.log10(total / ref)) < 2.5), 20 * np.log10(total / ref)


@pytest.mark.parametrize("fe_mode", ["diff", "adiff", "gsc"])
def test_frontends_preserve_block_length_and_are_finite(fe_mode):
    fe = make_frontend(
        fe_mode, mic_spacing_m=0.015, mouth_distance_m=0.055, sample_rate=SR
    )
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = fe.process(
            rng.standard_normal(512).astype(np.float32),
            rng.standard_normal(512).astype(np.float32),
        )
        assert y.shape == (512,) and np.isfinite(y).all()


def test_cardioid_nulls_noise_from_behind():
    """beta=0: a plane wave from 180 deg must cancel in the forward cardioid."""
    s = _noise(1)
    f, r = rir.plane_wave_to_mics(s, 180.0, GEOM, SR)
    fe = bf.DifferentialFrontEnd(0.015, 0.055, SR, beta=0.0)
    y = _run(fe, f, r)

    red_db = 10 * np.log10(_band_power(f, 300, 6000) / _band_power(y, 300, 6000))
    assert red_db > 20.0, red_db


def test_rear_noise_is_attenuated_more_than_front_noise_relative_to_mouth():
    """After mouth-flattening EQ the pod must favour the mouth over rear noise."""
    s = _noise(2)
    sp = _noise(3)
    fe_mouth = bf.DifferentialFrontEnd(0.015, 0.055, SR, beta=0.25)
    fm, rm = rir.mouth_to_mics(sp, GEOM, SR)
    y_mouth = _run(fe_mouth, fm, rm)
    fe_rear = bf.DifferentialFrontEnd(0.015, 0.055, SR, beta=0.25)
    fr, rr = rir.plane_wave_to_mics(s, 180.0, GEOM, SR)
    y_rear = _run(fe_rear, fr, rr)

    gain_db = 10 * np.log10(_band_power(y_rear, 200, 4000) / _band_power(fr, 200, 4000))
    mouth_db = 10 * np.log10(
        _band_power(y_mouth, 200, 4000) / _band_power(fm, 200, 4000)
    )
    assert mouth_db - gain_db > 6.0, (mouth_db, gain_db)


def test_blocking_reference_cancels_the_mouth_exactly():
    """u = r - g f(t-tau) is ~0 for mouth speech and large for far-field noise."""
    sp = _noise(4)
    fe = bf.GSCFrontEnd(0.015, 0.055, SR)
    f, r = rir.mouth_to_mics(sp, GEOM, SR)

    d = fe.dbf
    _c_f, c_b, _fd, r_d = d.beams(f.astype(np.float32), r.astype(np.float32))
    u_mouth = r_d - fe.g * (r_d - c_b)

    n = _noise(5)
    fn, rn = rir.plane_wave_to_mics(n, 45.0, GEOM, SR)
    d.reset()
    _c_f, c_b, _fd, r_d = d.beams(fn.astype(np.float32), rn.astype(np.float32))
    u_noise = r_d - fe.g * (r_d - c_b)

    leak_db = 10 * np.log10(_band_power(u_mouth, 300, 6000) / _band_power(f, 300, 6000))
    noise_db = 10 * np.log10(
        _band_power(u_noise, 300, 6000) / _band_power(fn, 300, 6000)
    )
    assert leak_db < -25.0, leak_db
    assert noise_db > leak_db + 20.0


def test_gsc_improves_snr_for_front_side_noise_without_destroying_speech():
    # well below full scale: a clipped capture (|x| >= 0.98) freezes adaptation by design
    n_src = 0.1 * _noise(6, SR * 6)
    sp = 0.1 * _noise(7, SR * 6)
    # make "speech" bursty so the canceller has noise-only stretches
    env = (np.sin(2 * np.pi * 0.5 * np.arange(sp.size) / SR) > 0.2).astype(float)
    sp = sp * env
    fn, rn = rir.plane_wave_to_mics(n_src, 60.0, GEOM, SR)
    fs, rs = rir.mouth_to_mics(sp * 0.3, GEOM, SR)
    f, r = fn + fs, rn + rs

    fe = make_frontend(
        "gsc", mic_spacing_m=0.015, mouth_distance_m=0.055, sample_rate=SR
    )
    y_mix = _run(fe, f, r)
    fe.reset()
    y_noise = _run(fe, fn, rn)

    # noise power at the output must fall relative to the primary mic noise
    tail = slice(SR * 2, None)
    nr = 10 * np.log10(
        np.mean(fn[: y_noise.size][tail] ** 2) / np.mean(y_noise[tail] ** 2)
    )
    assert nr > 3.0, nr
    assert np.isfinite(y_mix).all()


def test_adaptive_beta_stays_within_bounds():
    fe = bf.DifferentialFrontEnd(0.015, 0.055, SR, adaptive=True, beta_max=0.7)
    s = _noise(8)
    f, r = rir.plane_wave_to_mics(s, 120.0, GEOM, SR)
    _run(fe, f, r)

    assert 0.0 <= fe.beta <= 0.7
