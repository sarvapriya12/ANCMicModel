from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def _to_numpy(x: np.ndarray | list | tuple | object) -> np.ndarray:
    """Convert tensor, list, or array-like to 1-D float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    return np.squeeze(arr)


def calculate_erle(
    reference: np.ndarray | list | object,
    residual: np.ndarray | list | object,
    sample_rate: int = 16000,
    window_sec: float = 0.05,
    step_sec: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Echo Return Loss Enhancement (ERLE) over a rolling window.

    Matches math_sim_engine formulation:
        ERLE_dB = 10 * log10(P_reference / P_residual)

    Args:
        reference: Far-end echo signal or microphone input before cancellation.
        residual: Cleaned output/error signal after echo cancellation.
        sample_rate: Sampling frequency in Hz (default: 16000).
        window_sec: Duration of analysis window in seconds (default: 0.05 s = 50 ms)
                    or sample count if >= 1.
        step_sec: Hop size between consecutive windows in seconds (default: 0.025 s = 25 ms)
                  or sample count if >= 1.

    Returns:
        times: Array of window start timestamps in seconds.
        erle_db: Array of ERLE values in decibels for each window.
    """
    ref = _to_numpy(reference)
    res = _to_numpy(residual)

    if ref.ndim != 1 or res.ndim != 1:
        raise ValueError(f"Inputs must be 1-D audio signals, got shapes {ref.shape} and {res.shape}")

    min_len = min(len(ref), len(res))
    ref = ref[:min_len]
    res = res[:min_len]

    # Support both seconds and integer sample counts
    window = int(window_sec if window_sec >= 1.0 else sample_rate * window_sec)
    step = int(step_sec if step_sec >= 1.0 else sample_rate * step_sec)

    if window <= 0 or step <= 0:
        raise ValueError(f"Window and step must be positive integers, got window={window}, step={step}")

    if min_len <= window:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    erle_db = []
    times = []

    for i in range(0, min_len - window, step):
        reference_win = ref[i : i + window]
        residual_win = res[i : i + window]

        p_reference = np.mean(reference_win**2) + 1e-10
        p_residual = np.mean(residual_win**2) + 1e-10

        erle = 10.0 * np.log10(p_reference / p_residual)
        erle_db.append(erle)
        times.append(i / float(sample_rate))

    return np.array(times, dtype=np.float64), np.array(erle_db, dtype=np.float64)


def calculate_convergence_time(
    times: np.ndarray | list,
    erle_db: np.ndarray | list,
    threshold_db: float = 10.0,
    default: float = -1.0,
) -> float:
    """
    Calculate the convergence time (time in seconds to first reach the ERLE threshold).

    Matches math_sim_engine metric:
        Finds the first time t where erle[t] >= threshold_db.

    Args:
        times: Array of timestamps in seconds.
        erle_db: Array of corresponding ERLE values in dB.
        threshold_db: ERLE threshold in dB (default: 10.0 dB).
        default: Return value if threshold is never reached (default: -1.0).

    Returns:
        Time in seconds when threshold was first reached, or default (-1.0) if not achieved.
    """
    t_arr = _to_numpy(times)
    e_arr = _to_numpy(erle_db)

    if len(t_arr) == 0 or len(e_arr) == 0:
        return float(default)

    for t, e in zip(t_arr, e_arr):
        if e >= threshold_db:
            return float(t)

    return float(default)


def calculate_post_double_talk_erle(
    times: np.ndarray | list,
    erle_db: np.ndarray | list,
    start_time: float,
    end_time: float,
) -> float:
    """
    Calculate the average ERLE in dB over a specific post-double-talk time window.

    Matches math_sim_engine metric:
        post_dt_mask = (times >= start_time) & (times < end_time)
        post_dt_avg_erle = np.mean(erle[post_dt_mask])

    Args:
        times: Array of timestamps in seconds.
        erle_db: Array of ERLE values in dB.
        start_time: Window start time in seconds.
        end_time: Window end time in seconds.

    Returns:
        Average ERLE in dB within the specified interval, or 0.0 if empty.
    """
    t_arr = _to_numpy(times)
    e_arr = _to_numpy(erle_db)

    if len(t_arr) == 0 or len(e_arr) == 0:
        return 0.0

    mask = (t_arr >= start_time) & (t_arr < end_time)
    if not np.any(mask):
        return 0.0

    return float(np.mean(e_arr[mask]))


def calculate_si_sdr(
    reference: np.ndarray | list | object,
    estimate: np.ndarray | list | object,
    eps: float = 1e-10,
) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels.

    Args:
        reference: Clean target speech signal.
        estimate: Enhanced or processed speech estimate.
        eps: Numerical stability constant.

    Returns:
        SI-SDR in dB.
    """
    ref = _to_numpy(reference)
    est = _to_numpy(estimate)

    if ref.ndim != 1 or est.ndim != 1:
        raise ValueError(f"Inputs must be 1-D audio signals, got {ref.shape} and {est.shape}")

    min_len = min(len(ref), len(est))
    if min_len == 0:
        return 0.0

    ref = ref[:min_len]
    est = est[:min_len]

    # Zero-mean normalization
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    ref_energy = np.sum(ref**2)
    if ref_energy < eps:
        return 0.0

    # Orthogonal projection of estimate onto reference
    alpha = np.dot(est, ref) / (ref_energy + eps)
    e_target = alpha * ref
    e_noise = est - e_target

    target_energy = np.sum(ref**2)
    noise_energy = np.sum((e_noise / (np.abs(alpha) + eps))**2)

    si_sdr = 10.0 * np.log10((target_energy + eps) / (noise_energy + eps))
    return float(si_sdr)


def calculate_snr(
    reference: np.ndarray | list | object,
    estimate: np.ndarray | list | object,
    eps: float = 1e-10,
) -> float:
    """
    Calculate conventional Signal-to-Noise Ratio (SNR) in decibels.

    Formula:
        P_signal = mean(reference^2)
        P_noise = mean((reference - estimate)^2)
        SNR_dB = 10 * log10(P_signal / (P_noise + eps))

    Args:
        reference: Clean reference signal.
        estimate: Processed or noisy signal.
        eps: Numerical stability constant.

    Returns:
        SNR in dB.
    """
    ref = _to_numpy(reference)
    est = _to_numpy(estimate)

    if ref.ndim != 1 or est.ndim != 1:
        raise ValueError(f"Inputs must be 1-D audio signals, got {ref.shape} and {est.shape}")

    min_len = min(len(ref), len(est))
    if min_len == 0:
        return 0.0

    ref = ref[:min_len]
    est = est[:min_len]

    p_signal = np.mean(ref**2)
    p_noise = np.mean((ref - est) ** 2)

    snr = 10.0 * np.log10((p_signal + eps) / (p_noise + eps))
    return float(snr)


def calculate_coherence(
    primary: np.ndarray | list | object,
    reference: np.ndarray | list | object,
    nperseg: int = 256,
    fs: int = 16000,
) -> float:
    """
    Calculate the mean magnitude squared coherence between primary and reference signals.

    Matches Section 9 of AEC_ENGINEERING_REPORT:
        gamma[k] = |P_xd[k]|^2 / (P_x[k] * P_d[k] + eps)
        Constrained between 0.0 and 1.0.

    Args:
        primary: Primary microphone signal.
        reference: Reference microphone or far-end radio reference signal.
        nperseg: FFT segment length (default: 256).
        fs: Sampling rate in Hz (default: 16000).

    Returns:
        Mean magnitude-squared coherence (float in [0.0, 1.0]).
    """
    p = _to_numpy(primary)
    r = _to_numpy(reference)

    if p.ndim != 1 or r.ndim != 1:
        raise ValueError(f"Inputs must be 1-D audio signals, got {p.shape} and {r.shape}")

    min_len = min(len(p), len(r))
    if min_len < 16:
        return 0.0

    p = p[:min_len]
    r = r[:min_len]

    seg_len = min(nperseg, min_len)

    # Scipy coherence computes magnitude-squared coherence |P_xy|^2 / (P_xx * P_yy)
    _, cxy = sp_signal.coherence(p, r, fs=fs, nperseg=seg_len)

    valid_cxy = cxy[np.isfinite(cxy)]
    if len(valid_cxy) == 0:
        return 0.0

    mean_coherence = float(np.clip(np.mean(valid_cxy), 0.0, 1.0))
    return mean_coherence
