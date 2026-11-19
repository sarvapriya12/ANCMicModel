"""Capture <-> playback clock-drift compensation.

The I2S microphones are clocked by the Pi's I2S master, but the headphone path
(USB / Bluetooth / HDMI audio) runs on another oscillator. A 50 ppm mismatch
means one 512-sample block is lost or invented every ~3.5 minutes: an audible
click, forever. Instead of dropping or repeating samples we make the output
resampler's ratio track the FIFO fill with a PI controller (ratio changes by
tens of ppm, inaudible), costing ~2 multiply-adds per sample.
"""

from __future__ import annotations

import numpy as np


class DriftCompensator:
    """Elastic output FIFO with a PI-controlled linear-interpolation resampler."""

    def __init__(
        self,
        target_fill: int,
        capacity: int | None = None,
        kp_ppm_per_sample: float = 1.0,
        ki_ppm_per_sample_block: float = 0.001,
        max_ppm: float = 500.0,
        fill_smoothing: float = 0.02,
    ) -> None:
        """Gains follow from the plant: one ppm of ratio error moves the fill
        by ~block*1e-6 samples per block, so with kp = 1 ppm/sample a steady
        80 ppm drift costs only ~80 samples of fill error, and the integral
        term (ki) removes that over a few minutes. The fill is low-passed
        (``fill_smoothing`` per block) because the consumer drains it in whole
        blocks, which would otherwise look like +/-block noise to the loop."""
        self.target = int(target_fill)
        self.capacity = int(capacity if capacity is not None else 8 * target_fill)
        self.kp = kp_ppm_per_sample
        self.ki = ki_ppm_per_sample_block
        self.max_ppm = max_ppm
        self.alpha = fill_smoothing
        self.reset()

    def reset(self) -> None:
        self._fifo = np.zeros(0, dtype=np.float32)
        self._last = 0.0
        self._pos = 0.0  # fractional read position inside the next input block
        self._integral = 0.0
        self._fill_f = float(self.target)
        self.ppm = 0.0
        self.underruns = 0
        self.overruns = 0

    @property
    def fill(self) -> int:
        return int(self._fifo.size)

    def write(self, x: np.ndarray) -> None:
        """Resample a block with the current ratio and append it to the FIFO."""
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return
        self._fill_f += self.alpha * (self.fill - self._fill_f)
        err = self._fill_f - self.target
        self._integral = float(
            np.clip(self._integral + self.ki * err, -self.max_ppm, self.max_ppm)
        )
        self.ppm = float(
            np.clip(self.kp * err + self._integral, -self.max_ppm, self.max_ppm)
        )
        step = 1.0 + self.ppm * 1e-6  # input samples consumed per output sample

        ext = np.concatenate((np.array([self._last], dtype=np.float32), x))
        # positions measured in ``ext`` coordinates (index 0 == previous last sample)
        m = max(int(np.floor((ext.size - 1 - self._pos) / step)) + 1, 0)
        pos = self._pos + step * np.arange(m)
        pos = pos[pos <= ext.size - 1]
        out = np.interp(pos, np.arange(ext.size), ext).astype(np.float32)
        consumed = (pos[-1] + step) if pos.size else self._pos
        self._pos = consumed - (ext.size - 1)
        self._last = float(x[-1])

        self._fifo = np.concatenate((self._fifo, out))
        if self._fifo.size > self.capacity:
            self.overruns += 1
            self._fifo = self._fifo[-self.capacity :]

    def read(self, n: int) -> np.ndarray:
        """Exactly ``n`` samples; zero-fills (and counts an underrun) if short."""
        if self._fifo.size >= n:
            out, self._fifo = self._fifo[:n], self._fifo[n:]
            return out
        self.underruns += 1
        out = np.zeros(n, dtype=np.float32)
        out[: self._fifo.size] = self._fifo
        self._fifo = self._fifo[:0]
        return out
