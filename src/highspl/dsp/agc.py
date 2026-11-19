"""Level management: streaming AGC (before the neural stage) and a lookahead
hearing-safety limiter (after it).

Why: the pretrained enhancer was trained on mixtures at about -25..-10 dBFS
(see models/fastenhancer/b/config.yaml: ``noisy_rms_range``). A quiet INMP441
feeds it much less (it then over-suppresses speech), a 130 dB scene feeds it
full-scale. The legacy pipeline had no level control at all, only a flat x0.8.

Both stages are block-vectorised: a handful of operations per sample.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-12


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


class StreamingAGC:
    """Slow, gain-bounded AGC with impulse protection.

    * Level is the RMS of each block, tracked with fast attack / slow release.
    * Gain moves towards ``target_dbfs - level`` at a limited slew rate
      (``attack_db_s`` when reducing, ``release_db_s`` when raising) and is
      interpolated linearly across each block (no zipper noise).
    * Blocks whose crest factor shows an impulse (gunshot / blast) do **not**
      raise the level estimate and hold the gain, so the AGC does not pump
      after every bang.
    * Digital silence cannot drive the gain to its maximum: the level floor
      ``floor_dbfs`` caps the boost.
    """

    def __init__(
        self,
        target_dbfs: float = -24.0,
        max_gain_db: float = 24.0,
        min_gain_db: float = -24.0,
        attack_db_s: float = 40.0,
        release_db_s: float = 6.0,
        floor_dbfs: float = -70.0,
        impulse_crest_db: float = 18.0,
        sample_rate: int = 48_000,
    ) -> None:
        self.target = target_dbfs
        self.max_gain = max_gain_db
        self.min_gain = min_gain_db
        self.attack = attack_db_s
        self.release = release_db_s
        self.floor = floor_dbfs
        self.impulse_crest = impulse_crest_db
        self.fs = sample_rate
        self.reset()

    def reset(self) -> None:
        self.gain_db = 0.0
        self.level_db = self.target
        self.last_gain = np.ones(
            0, dtype=np.float32
        )  # per-sample gain of the last block

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        if n == 0:
            return x
        power = float(np.mean(x.astype(np.float64) ** 2)) + EPS
        peak = float(np.max(np.abs(x))) + EPS
        level_db = max(10.0 * np.log10(power), self.floor)
        crest_db = 20.0 * np.log10(peak / np.sqrt(power))
        dt = n / self.fs

        if crest_db > self.impulse_crest:
            new_gain = self.gain_db  # impulse: hold the gain
        else:
            # one-pole level tracker (fast attack, slow release)
            a = 0.5 if level_db > self.level_db else 0.05
            self.level_db += a * (level_db - self.level_db)
            want = float(
                np.clip(self.target - self.level_db, self.min_gain, self.max_gain)
            )
            step = (self.release if want > self.gain_db else self.attack) * dt
            new_gain = self.gain_db + float(np.clip(want - self.gain_db, -step, step))

        ramp = np.linspace(self.gain_db, new_gain, n, endpoint=False)
        self.gain_db = new_gain
        self.last_gain = (10.0 ** (ramp / 20.0)).astype(np.float32)
        return (x * self.last_gain).astype(np.float32)


class LookaheadLimiter:
    """Hearing-safety peak limiter with a short lookahead and hold.

    For output sample m (the input delayed by S samples)::

        T[i] = min(req[i - w + 1 .. i])          w = hold + S   (trailing min)
        G[m] = mean(T[m .. m + S - 1])           (forward moving average)

    with ``req[n] = min(1, ceiling / |x[n]|)``. Every T in G's average covers
    sample m, so ``G[m] <= req[m]`` and ``|y| <= ceiling`` is *guaranteed*, while
    the average makes the gain ramp down over S samples *before* the peak
    arrives (no clicks) and the hold keeps it low for ``hold`` samples after.
    """

    def __init__(
        self,
        ceiling_dbfs: float = -1.0,
        lookahead: int = 48,
        hold: int = 480,
    ) -> None:
        self.ceiling = db_to_lin(ceiling_dbfs)
        self.S = int(lookahead)
        self.H = int(hold)
        self.w = self.H + self.S
        self.shift = (self.w - 1) // 2
        self.reset()

    @property
    def latency(self) -> int:
        return self.S

    def set_ceiling(self, ceiling_lin: float) -> None:
        self.ceiling = float(ceiling_lin)

    def reset(self) -> None:
        self._audio_hist = np.zeros(self.S, dtype=np.float32)
        self._req_hist = np.ones(self.H + 2 * self.S, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n = x.size
        req = np.minimum(1.0, self.ceiling / (np.abs(x) + EPS)).astype(np.float32)

        ext_req = np.concatenate((self._req_hist, req))
        hh = self._req_hist.size
        centre_min = ndimage.minimum_filter1d(ext_req, size=self.w, mode="nearest")
        # forward moving average of the trailing min, expressed on the centred array
        cm = np.convolve(
            centre_min, np.full(self.S, 1.0 / self.S, dtype=np.float32), mode="valid"
        )
        first = hh - self.S - self.shift
        g = cm[first : first + n]

        ext_audio = np.concatenate((self._audio_hist, x))
        y = ext_audio[:n] * g

        self._audio_hist = ext_audio[n : n + self.S].copy()
        self._req_hist = ext_req[-(self.H + 2 * self.S) :]
        return y.astype(np.float32)
