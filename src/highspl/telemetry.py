"""Live telemetry that does not lie.

The legacy lab UI showed "noise reduction" as 10*log10(output energy / input
energy), which counts speech as noise (muting everything scored "perfect") and
a latency of ``compute + 10.6`` with the 10.6 hard-coded. Replacements:

* :class:`NoiseFloorTracker` - minimum-statistics noise floor of a stream. The
  floor of the output vs the floor of the input is a *noise-in-pauses*
  reduction that needs no ground truth and cannot be gamed by muting speech
  (a muted output has a floor but also zero speech; see ``speech_level``).
* :func:`latency_budget` - the latency actually implied by the configuration.
"""

from __future__ import annotations

from collections import deque

import numpy as np

EPS = 1e-12


class NoiseFloorTracker:
    """Rolling low-percentile block power (dBFS): noise floor in speech pauses
    and, as the high percentile, the speech/active level."""

    def __init__(
        self, seconds: float = 4.0, block: int = 512, sr: int = 48_000
    ) -> None:
        self._p = deque(maxlen=max(int(seconds * sr / block), 8))

    def update(self, x: np.ndarray) -> None:
        # a single NaN/inf must not poison the percentile for the next seconds
        x = np.nan_to_num(
            np.asarray(x, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0
        )
        self._p.append(float(np.mean(x**2)) + EPS)

    def _pct(self, q: float) -> float:
        if len(self._p) < 8:
            return float("nan")
        return 10.0 * np.log10(float(np.percentile(np.fromiter(self._p, float), q)))

    @property
    def floor_dbfs(self) -> float:
        return self._pct(10.0)

    @property
    def speech_level_dbfs(self) -> float:
        return self._pct(90.0)

    def reset(self) -> None:
        self._p.clear()


def latency_budget(
    block: int,
    sr: int = 48_000,
    model_latency_samples: int = 512,
    limiter_samples: int = 0,
    capture_periods: float = 1.0,
    playback_periods: float = 1.0,
    compute_ms: float = 0.0,
) -> dict[str, float]:
    """Latency implied by the configuration, in milliseconds.

    ``model_latency_samples`` is the algorithmic delay of the neural stage
    *measured* by the evaluator (512 samples = 10.67 ms for FastEnhancer-B);
    OS / driver buffering is not included, so the total is a lower bound."""
    ms = 1000.0 / sr
    parts = {
        "capture_block_ms": capture_periods * block * ms,
        "model_ms": model_latency_samples * ms,
        "limiter_ms": limiter_samples * ms,
        "compute_ms": compute_ms,
        "playback_block_ms": playback_periods * block * ms,
    }
    parts["total_lower_bound_ms"] = sum(parts.values())
    return parts
