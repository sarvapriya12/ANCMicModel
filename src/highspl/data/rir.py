"""Geometry and propagation for the two-microphone pod.

Coordinate system (matches the physical build):

    rear mic (reference)   front mic (primary)         mouth
         z = -d                  z = 0                z = +r1  ──►

The primary mic faces the lips, the reference mic sits directly behind it
facing away: a back-to-back, near-field *endfire* pair of spacing ``d``.
A far-field source at polar angle ``theta`` (0° = in front of the user, 180° =
behind) reaches the front mic ``d*cos(theta)/c`` before the rear mic.

All signals are in pascals; hardware scaling happens in ``data.clipping``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

C_SOUND = 343.0  # m/s


@dataclass(frozen=True)
class PodGeometry:
    spacing_m: float = 0.015  # mic-to-mic distance d
    mouth_m: float = 0.055  # lips to the primary mic r1
    shadow_db: float = 3.0  # HF attenuation of the housing on the far-side mic

    @property
    def mouth_gain(self) -> float:
        """Amplitude of mouth speech at the rear mic relative to the front mic
        (spherical spreading only): r1 / (r1 + d)."""
        return self.mouth_m / (self.mouth_m + self.spacing_m)

    def delay_samples(self, sr: int) -> float:
        """Inter-mic travel time of an on-axis wave, in samples."""
        return self.spacing_m / C_SOUND * sr


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay (or advance, if negative) ``x`` by a fractional number of samples
    using an exact frequency-domain linear phase. Output length == input."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or delay_samples == 0.0:
        return x.copy()
    pad = int(np.ceil(abs(delay_samples))) + 256
    nfft = 1 << int(np.ceil(np.log2(n + 2 * pad)))
    spec = np.fft.rfft(x, nfft)
    k = np.arange(spec.size)
    spec *= np.exp(-2j * np.pi * k * delay_samples / nfft)
    return np.fft.irfft(spec, nfft)[:n]


def _rbj_high_shelf(sr: int, f0: float, gain_db: float) -> np.ndarray:
    """RBJ cookbook high-shelf biquad (second-order section)."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cosw, sinw = np.cos(w0), np.sin(w0)
    alpha = sinw / 2.0 * np.sqrt(2.0)  # shelf slope S = 1
    sq = 2.0 * np.sqrt(a_lin) * alpha
    b0 = a_lin * ((a_lin + 1) + (a_lin - 1) * cosw + sq)
    b1 = -2 * a_lin * ((a_lin - 1) + (a_lin + 1) * cosw)
    b2 = a_lin * ((a_lin + 1) + (a_lin - 1) * cosw - sq)
    a0 = (a_lin + 1) - (a_lin - 1) * cosw + sq
    a1 = 2 * ((a_lin - 1) - (a_lin + 1) * cosw)
    a2 = (a_lin + 1) - (a_lin - 1) * cosw - sq
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def high_shelf(
    x: np.ndarray, gain_db: float, sr: int, f0: float = 2000.0
) -> np.ndarray:
    """Apply a high-shelf gain (negative = cut) above ~f0."""
    if abs(gain_db) < 1e-6:
        return np.asarray(x, dtype=np.float64).copy()
    return sps.sosfilt(_rbj_high_shelf(sr, f0, gain_db), np.asarray(x, np.float64))


def mouth_to_mics(
    speech_pa: np.ndarray, geom: PodGeometry, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Near-field mouth speech at (front, rear). ``speech_pa`` is the pressure
    at the *front* mic; the rear mic sees it later, weaker and HF-shadowed."""
    front = np.asarray(speech_pa, dtype=np.float64).copy()
    rear = geom.mouth_gain * fractional_delay(front, geom.delay_samples(sr))
    rear = high_shelf(rear, -geom.shadow_db, sr)
    return front, rear


def plane_wave_to_mics(
    sig_pa: np.ndarray, theta_deg: float, geom: PodGeometry, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Far-field plane wave from polar angle ``theta_deg`` at (front, rear).

    ``sig_pa`` is the pressure at the *pod centre-front mic* reference point.
    The mic on the far side of the pod is HF-shadowed in proportion to how
    directly the wave hits the other mic."""
    cos_t = float(np.cos(np.deg2rad(theta_deg)))
    sig = np.asarray(sig_pa, dtype=np.float64)
    front = sig.copy()
    rear = fractional_delay(sig, geom.delay_samples(sr) * cos_t)
    front = high_shelf(front, -geom.shadow_db * max(0.0, -cos_t), sr)
    rear = high_shelf(rear, -geom.shadow_db * max(0.0, cos_t), sr)
    return front, rear


def diffuse_coherence(freqs_hz: np.ndarray, spacing_m: float) -> np.ndarray:
    """Theoretical coherence of a spherically isotropic field between two
    omnidirectional mics: sinc(2*pi*f*d/c)."""
    return np.sinc(2.0 * np.asarray(freqs_hz) * spacing_m / C_SOUND)


def diffuse_pair(
    noise_a: np.ndarray,
    noise_b: np.ndarray,
    spacing_m: float,
    sr: int,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn two *independent* realisations of the same noise process into a
    pair with the diffuse-field coherence between them (STFT mixing).

    out1 = a ;  out2 = G*a + sqrt(1-G^2)*b, per frequency, G = sinc(2*pi*f*d/c).
    Both outputs keep the PSD of the input noise."""
    n = min(noise_a.size, noise_b.size)
    a = np.asarray(noise_a[:n], dtype=np.float64)
    b = np.asarray(noise_b[:n], dtype=np.float64)
    f, _, spec_a = sps.stft(a, fs=sr, nperseg=nperseg)
    _, _, spec_b = sps.stft(b, fs=sr, nperseg=nperseg)
    gamma = diffuse_coherence(f, spacing_m)[:, None]
    spec_2 = gamma * spec_a + np.sqrt(np.clip(1.0 - gamma**2, 0.0, 1.0)) * spec_b
    _, out1 = sps.istft(spec_a, fs=sr, nperseg=nperseg)
    _, out2 = sps.istft(spec_2, fs=sr, nperseg=nperseg)
    return out1[:n], out2[:n]
