import numpy as np
import pytest
import torch

from highspl.evaluation.metrics import (
    calculate_coherence,
    calculate_convergence_time,
    calculate_erle,
    calculate_post_double_talk_erle,
    calculate_si_sdr,
    calculate_snr,
)


def test_calculate_erle_known_attenuation():
    fs = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    reference = np.sin(2 * np.pi * 500 * t).astype(np.float64)

    # Residual attenuated by factor of 10 in amplitude (20 dB in power)
    residual = (reference * 0.1).astype(np.float64)

    times, erle_db = calculate_erle(
        reference,
        residual,
        sample_rate=fs,
        window_sec=0.05,
        step_sec=0.025,
    )

    assert len(times) > 0
    assert len(times) == len(erle_db)
    # ERLE should be approximately 20 dB across windows
    assert np.allclose(erle_db, 20.0, atol=1.0)


def test_calculate_erle_with_torch_tensor():
    fs = 16000
    ref = torch.randn(8000)
    res = ref * 0.01  # 40 dB attenuation

    times, erle_db = calculate_erle(ref, res, sample_rate=fs)
    assert isinstance(times, np.ndarray)
    assert isinstance(erle_db, np.ndarray)
    assert len(erle_db) > 0
    assert np.all(erle_db > 30.0)


def test_calculate_erle_short_input():
    # Signal shorter than window size (50ms @ 16kHz = 800 samples)
    ref = np.zeros(200)
    res = np.zeros(200)

    times, erle_db = calculate_erle(ref, res, sample_rate=16000, window_sec=0.05)
    assert len(times) == 0
    assert len(erle_db) == 0


def test_calculate_convergence_time():
    times = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
    erle_db = np.array([2.0, 5.0, 8.5, 11.2, 14.0, 18.0])

    # Should first reach >= 10 dB at index 3 (time 0.15 s)
    t_conv = calculate_convergence_time(times, erle_db, threshold_db=10.0)
    assert t_conv == pytest.approx(0.15)


def test_calculate_convergence_time_not_reached():
    times = np.array([0.0, 0.1, 0.2])
    erle_db = np.array([1.0, 3.0, 5.0])

    t_conv = calculate_convergence_time(times, erle_db, threshold_db=10.0)
    assert t_conv == -1.0

    # Custom default
    t_conv_none = calculate_convergence_time(
        times, erle_db, threshold_db=10.0, default=-999.0
    )
    assert t_conv_none == -999.0


def test_calculate_post_double_talk_erle():
    times = np.array([0.0, 1.0, 2.0, 2.7, 2.9, 3.1, 4.0])
    erle_db = np.array([10.0, 12.0, 14.0, 20.0, 22.0, 24.0, 15.0])

    # Window from 2.6 to 3.4 includes times [2.7, 2.9, 3.1] with values [20.0, 22.0, 24.0]
    avg_erle = calculate_post_double_talk_erle(
        times, erle_db, start_time=2.6, end_time=3.4
    )
    expected_avg = np.mean([20.0, 22.0, 24.0])
    assert avg_erle == pytest.approx(expected_avg)

    # Empty window
    avg_empty = calculate_post_double_talk_erle(
        times, erle_db, start_time=10.0, end_time=12.0
    )
    assert avg_empty == 0.0


def test_calculate_si_sdr():
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 1, 16000)

    # Identical signal: SI-SDR should be extremely high (> 70 dB)
    si_sdr_clean = calculate_si_sdr(clean, clean)
    assert si_sdr_clean > 70.0

    # Scale invariance: estimate is 4.5 * clean
    scaled_clean = clean * 4.5
    si_sdr_scaled = calculate_si_sdr(clean, scaled_clean)
    assert si_sdr_scaled > 70.0
    assert abs(si_sdr_clean - si_sdr_scaled) < 1.0

    # Uncorrelated noise: SI-SDR should be very low / negative
    noise = rng.normal(0, 1, 16000)
    si_sdr_noise = calculate_si_sdr(clean, noise)
    assert si_sdr_noise < 5.0


def test_calculate_snr():
    rng = np.random.default_rng(123)
    clean = rng.normal(0, 1, 16000)

    # Perfect match: SNR very high
    snr_clean = calculate_snr(clean, clean)
    assert snr_clean > 70.0

    # Noise added with 20 dB SNR
    noise = rng.normal(0, 0.1, 16000)  # std 0.1 vs 1.0 -> 20 dB
    snr_noisy = calculate_snr(clean, clean + noise)
    assert snr_noisy == pytest.approx(20.0, abs=1.5)


def test_calculate_coherence():
    rng = np.random.default_rng(999)
    # Strongly coherent signals
    sig = rng.normal(0, 1, 8000)
    coh_identical = calculate_coherence(sig, sig, nperseg=256)
    assert coh_identical == pytest.approx(1.0, abs=0.01)

    # Correlated with filtered / scaled version
    scaled_sig = sig * 0.5
    coh_scaled = calculate_coherence(sig, scaled_sig, nperseg=256)
    assert coh_scaled == pytest.approx(1.0, abs=0.01)

    # Uncorrelated white noise
    noise1 = rng.normal(0, 1, 16000)
    noise2 = rng.normal(0, 1, 16000)
    coh_noise = calculate_coherence(noise1, noise2, nperseg=256)
    assert 0.0 <= coh_noise < 0.25
