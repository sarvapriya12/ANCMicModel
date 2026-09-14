import numpy as np


class PeakLimiter:
    def __init__(self, ceiling_db: float) -> None:
        if not np.isfinite(ceiling_db):
            raise ValueError("invalid ceiling")
        self.ceiling_db = float(ceiling_db)
        self.ceiling = 10.0 ** (self.ceiling_db / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError("input must be a numpy array")
        if not np.all(np.isfinite(audio)):
            raise ValueError("input contains non-finite values")

        peak = np.max(np.abs(audio))
        if peak > self.ceiling:
            ratio = self.ceiling / peak
        else:
            ratio = 1.0

        output = (audio * ratio).astype(np.float32)
        return output
