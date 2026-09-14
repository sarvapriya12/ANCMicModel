from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelState:
    """State owned by a streaming model backend."""

    backend: object


class StreamingEnhancer(ABC):
    """Common streaming interface for all speech-enhancement models."""

    sample_rate_in: int
    sample_rate_out: int

    @abstractmethod
    def reset(self) -> ModelState:
        """Create/reset the model streaming state."""
        raise NotImplementedError

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        state: ModelState,
    ) -> tuple[np.ndarray, ModelState]:
        """Process one audio chunk while preserving model state."""
        raise NotImplementedError

    @abstractmethod
    def flush(
        self,
        state: ModelState,
    ) -> tuple[np.ndarray, ModelState]:
        """Flush any audio buffered internally by the model."""
        raise NotImplementedError
