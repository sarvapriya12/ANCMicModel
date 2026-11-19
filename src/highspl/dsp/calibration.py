"""Microphone-pair calibration.

With a ~2-sample inter-mic delay and a 0.79 mouth gain, the differential and
GSC front-ends are only as good as the match between the two capsules:

* gain mismatch (INMP441 sensitivity is +/-1 dB) leaks speech into the GSC's
  noise reference and ruins the cardioid's rear null;
* the I2S L/R slot offset (up to ~1 sample) shifts the apparent inter-mic delay.

Both are *properties of the device* and show up in the transfer function from
the front to the rear mic for **near-field mouth speech**: r ~ g_eff * f(t-tau_eff).
We estimate (g_eff, tau_eff) from clean speech (run once in a quiet place, or
on speech-dominant segments) and correct the rear channel to the nominal pod
geometry (g_nom, tau_nom).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .beamformer import DELAY, N_TAPS, fractional_delay_fir


def estimate_mouth_path(
    front: np.ndarray,
    rear: np.ndarray,
    fs: int = 48_000,
    band: tuple[float, float] = (200.0, 3000.0),
    nperseg: int = 1024,
    min_coherence: float = 0.9,
) -> dict[str, float]:
    """Estimate ``rear ~ gain * front(t - delay)`` from speech-dominant audio.

    Returns gain (linear), delay (samples) and the mean coherence used.
    Cross-spectral estimate: H = csd(f, r) / psd(f); gain = |H| and
    delay = -d(phase)/d(omega), both over bins with high coherence."""
    f, pff = sps.welch(front, fs, nperseg=nperseg)
    _, prr = sps.welch(rear, fs, nperseg=nperseg)
    _, pfr = sps.csd(front, rear, fs, nperseg=nperseg)
    coh = np.abs(pfr) ** 2 / (pff * prr + 1e-30)
    sel = (f >= band[0]) & (f <= band[1]) & (coh >= min_coherence)
    if sel.sum() < 4:
        return {
            "gain": float("nan"),
            "delay": float("nan"),
            "coherence": float(coh[(f >= band[0]) & (f <= band[1])].mean()),
        }
    h = pfr[sel] / (pff[sel] + 1e-30)
    w = coh[sel] * pff[sel]
    gain = float(np.sum(np.abs(h) * w) / np.sum(w))
    omega = 2.0 * np.pi * f[sel] / fs  # rad/sample
    phase = np.unwrap(np.angle(h))
    # weighted least-squares line through the origin: phase = -delay * omega
    delay = float(-np.sum(w * omega * phase) / np.sum(w * omega**2))
    return {"gain": gain, "delay": delay, "coherence": float(coh[sel].mean())}


class RearCalibrator:
    """Corrects the rear channel so that mouth speech matches the nominal pod:
    ``r' = r * (g_nom / g_eff)`` delayed by ``tau_nom - tau_eff`` (fractional).

    Both channels leave delayed by ``DELAY`` samples so the correction can be
    non-causal. Disabled == identity with no delay."""

    def __init__(self, nominal_gain: float, nominal_delay: float) -> None:
        self.g_nom = nominal_gain
        self.tau_nom = nominal_delay
        self.enabled = False
        self.gain_corr = 1.0
        self.delay_corr = 0.0
        self.reset()

    @property
    def latency(self) -> int:
        return DELAY if self.enabled else 0

    def set_from_measurement(self, gain_eff: float, delay_eff: float) -> None:
        if not (np.isfinite(gain_eff) and np.isfinite(delay_eff)) or gain_eff <= 0:
            self.enabled = False
            return
        self.gain_corr = float(np.clip(self.g_nom / gain_eff, 0.5, 2.0))
        # correction in [-1.5, 1.5] samples keeps the 19-tap FIR accurate
        self.delay_corr = float(np.clip(self.tau_nom - delay_eff, -1.5, 1.5))
        self._fir = fractional_delay_fir(self.delay_corr)
        self.enabled = True
        self.reset()

    def reset(self) -> None:
        self._zi = np.zeros(N_TAPS - 1)
        self._hist = np.zeros(DELAY)

    def process(
        self, front: np.ndarray, rear: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return front, rear
        r, self._zi = sps.lfilter(
            self._fir, [1.0], rear.astype(np.float64), zi=self._zi
        )
        ext = np.concatenate((self._hist, front.astype(np.float64)))
        self._hist = ext[-DELAY:]
        return ext[: front.size].astype(np.float32), (self.gain_corr * r).astype(
            np.float32
        )
