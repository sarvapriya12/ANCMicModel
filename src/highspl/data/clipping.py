"""Hardware model: acoustic pressure in -> what the Pi actually receives.

Build: two I2S MEMS microphones (INMP441-class) wired straight to the
Raspberry Pi 5's I2S port (shared BCLK/LRCLK/data), captured by ALSA. There is
no microcontroller and no USB hop in the signal path.

Chain, in order:

  1. sensitivity: pascals -> full-scale units, +/- per-mic gain mismatch
  2. ~60 Hz high-pass (the mic's internal HPF)
  3. acoustic/electrical overload: soft knee, then hard clip at full scale.
     Full scale == the mic's AOP (120 dB SPL for the INMP441)
  4. self-noise (INMP441: 61 dB SNR == 33 dB(A) SPL equivalent; modelled white)
  5. ~15 kHz low-pass (internal anti-alias/decimation response)
  6. I2S L/R sampling offset: the reference (right) slot trails by 0..1 sample
  7. 24-bit quantisation (INMP441 data is 24-bit, delivered in a 32-bit slot)
  8. (optional) software DC blocker; off by default, the mic's own ~60 Hz
     high-pass already removes DC
  9. ALSA overrun: whole capture periods (both channels together) are lost
     and zero-filled. Rare on a healthy system, but real when the Pi is busy

Stages 1-2 and 5-8 are linear/benign; 3 and 9 are the non-linear failure
modes the pipeline must survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .rir import fractional_delay

P_REF = 20e-6  # Pa, 0 dB SPL


@dataclass(frozen=True)
class MicHardware:
    aop_db: float = 120.0
    self_noise_db_spl: float = 33.0
    mismatch_db: tuple[float, float] = (0.0, 0.0)  # (front, rear) gain offsets
    hpf_hz: float = 60.0
    lpf_hz: float = 15000.0
    i2s_offset_samples: float = 0.0
    dc_blocker: float = 0.0  # 0 = off (no firmware stage on this build)
    soft_knee: float = 0.8
    dropout_period_samples: int = 512  # ALSA period size
    dropout_prob: float = 0.0  # probability a period is lost
    bits: int = 24

    @property
    def pa_full_scale(self) -> float:
        """Peak pressure (Pa) mapping to digital full scale: a sine of rms
        ``aop_db`` has this peak."""
        return P_REF * 10.0 ** (self.aop_db / 20.0) * np.sqrt(2.0)


def spl_to_pa_rms(db_spl: float) -> float:
    return P_REF * 10.0 ** (db_spl / 20.0)


def pa_rms_to_spl(pa_rms: float) -> float:
    return 20.0 * np.log10(max(pa_rms, 1e-30) / P_REF)


def soft_clip(x: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Linear up to ``knee``; tanh compression to a hard ceiling of 1.0 above."""
    ax = np.abs(x)
    over = ax > knee
    out = x.copy()
    if over.any():
        span = 1.0 - knee
        out[over] = np.sign(x[over]) * (knee + span * np.tanh((ax[over] - knee) / span))
    return np.clip(out, -1.0, 1.0)


def _hpf(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    if fc <= 0:
        return x
    return sps.sosfilt(sps.butter(2, fc, "high", fs=sr, output="sos"), x)


def _lpf(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    if fc >= 0.5 * sr:
        return x
    return sps.sosfilt(sps.butter(4, fc, "low", fs=sr, output="sos"), x)


def _dc_block(x: np.ndarray, r: float) -> np.ndarray:
    if r <= 0:
        return x
    return sps.lfilter([1.0, -1.0], [1.0, -r], x)


def apply_linear_chain(
    x_pa: np.ndarray, hw: MicHardware, sr: int, mic: str = "front"
) -> np.ndarray:
    """Only the linear parts of the chain (no overload, noise, quantisation or
    loss). This is what a *perfect* denoiser would output for clean speech."""
    gain = 10.0 ** ((hw.mismatch_db[0 if mic == "front" else 1]) / 20.0)
    x = np.asarray(x_pa, dtype=np.float64) / hw.pa_full_scale * gain
    x = _hpf(x, hw.hpf_hz, sr)
    x = _lpf(x, hw.lpf_hz, sr)
    return _dc_block(x, hw.dc_blocker)


def apply_chain(
    front_pa: np.ndarray,
    rear_pa: np.ndarray,
    hw: MicHardware,
    sr: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run both mic channels through the hardware model.

    Returns ``(front, rear, info)``; ``front``/``rear`` are float32 in [-1, 1)
    on the 24-bit grid; ``info`` holds clip fractions and the loss mask."""
    n = min(front_pa.size, rear_pa.size)
    chans = []
    clip_frac = []
    for pa, g_db in zip((front_pa[:n], rear_pa[:n]), hw.mismatch_db):
        x = np.asarray(pa, dtype=np.float64) / hw.pa_full_scale * 10.0 ** (g_db / 20.0)
        x = _hpf(x, hw.hpf_hz, sr)
        pre = np.abs(x)
        x = soft_clip(x, hw.soft_knee)
        clip_frac.append(float(np.mean(pre >= 1.0)))
        noise_rms = spl_to_pa_rms(hw.self_noise_db_spl) / hw.pa_full_scale
        x = x + noise_rms * rng.standard_normal(n)
        x = _lpf(x, hw.lpf_hz, sr)
        chans.append(x)

    front, rear = chans
    if hw.i2s_offset_samples:
        rear = fractional_delay(rear, hw.i2s_offset_samples)

    scale = float(2 ** (hw.bits - 1))
    lo, hi = -1.0, (scale - 1.0) / scale
    front = np.clip(np.round(front * scale) / scale, lo, hi)
    rear = np.clip(np.round(rear * scale) / scale, lo, hi)

    if hw.dc_blocker > 0.0:
        # A software DC blocker works in the integer domain: re-quantise.
        front = np.clip(
            np.round(_dc_block(front, hw.dc_blocker) * scale) / scale, lo, hi
        )
        rear = np.clip(np.round(_dc_block(rear, hw.dc_blocker) * scale) / scale, lo, hi)

    lost = np.zeros(n, dtype=bool)
    if hw.dropout_prob > 0.0:
        per = hw.dropout_period_samples
        n_pk = int(np.ceil(n / per))
        drop = rng.random(n_pk) < hw.dropout_prob
        for k in np.nonzero(drop)[0]:
            lost[k * per : (k + 1) * per] = True
        front = np.where(lost, 0.0, front)
        rear = np.where(lost, 0.0, rear)

    info = {
        "clip_fraction_front": clip_frac[0],
        "clip_fraction_rear": clip_frac[1],
        "lost_fraction": float(lost.mean()),
        "lost_mask": lost,
    }
    return front.astype(np.float32), rear.astype(np.float32), info


def drift_resample(x: np.ndarray, ppm: float) -> np.ndarray:
    """Model a clock running ``ppm`` parts-per-million fast/slow relative to the
    nominal rate (linear interpolation; used by the I/O drift tests)."""
    x = np.asarray(x, dtype=np.float64)
    ratio = 1.0 + ppm * 1e-6
    t = np.arange(int(x.size / ratio)) * ratio
    return np.interp(t, np.arange(x.size), x)
