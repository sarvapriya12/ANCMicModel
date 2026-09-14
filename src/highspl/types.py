from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

Audio = np.ndarray


@dataclass
class AudioChunk:
    """A chunk of audio flowing through the streaming pipeline."""

    samples: Audio
    sample_rate: int
    timestamp_samples: int = 0
    is_last: bool = False


@dataclass
class StreamState:
    """Mutable state carried between streaming process calls."""

    payload: dict[str, Any] = field(default_factory=dict)


class StreamProcessor(Protocol):
    """Protocol implemented by streaming DSP/model processors."""

    def reset(self) -> StreamState: ...

    def process(
        self,
        chunk: AudioChunk,
        state: StreamState,
    ) -> tuple[AudioChunk, StreamState]: ...

    def flush(
        self,
        state: StreamState,
    ) -> tuple[AudioChunk, StreamState]: ...


class Enhancer(Protocol):
    """Protocol implemented by speech-enhancement backends."""

    def reset(self) -> StreamState: ...

    def process(
        self,
        chunk: AudioChunk,
        state: StreamState,
    ) -> tuple[AudioChunk, StreamState]: ...

    def flush(
        self,
        state: StreamState,
    ) -> tuple[AudioChunk, StreamState]: ...
