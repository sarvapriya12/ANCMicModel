import numpy as np


class Gain:
    def __init__(self, db: float) -> None:
        if not np.isfinite(db):
            raise ValueError("invalid gain")
        self.db = float(db)
        self.multiplier = 10.0 ** (self.db / 20.0)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError("input must be a numpy array")
        if not np.all(np.isfinite(audio)):
            raise ValueError("input contains non-finite values")

        output = (audio * self.multiplier).astype(np.float32)
        return output
