from dataclasses import dataclass

import numpy as np
import soxr


@dataclass
class ResamplerState:
    """State carried between streaming resampler calls."""

    samples_processed: int = 0


class StatefulResampler:
    """Streaming sample-rate converter using libsoxr via python-soxr.

    Designed for the High-SPL pipeline's 32 kHz <-> 48 kHz conversion.

    The resampler itself is stateful inside ``soxr.ResampleStream``.
    ``ResamplerState`` additionally tracks the number of input samples
    processed by the application-level stream.
    """

    def __init__(
        self,
        input_rate_hz: int,
        output_rate_hz: int,
        channels: int = 1,
        quality: str = "HQ",
    ) -> None:
        if input_rate_hz <= 0:
            raise ValueError("input_rate_hz must be positive")

        if output_rate_hz <= 0:
            raise ValueError("output_rate_hz must be positive")

        if channels <= 0:
            raise ValueError("channels must be positive")

        if quality not in {"QQ", "LQ", "MQ", "HQ", "VHQ"}:
            raise ValueError("quality must be one of: QQ, LQ, MQ, HQ, VHQ")

        self.input_rate_hz = input_rate_hz
        self.output_rate_hz = output_rate_hz
        self.channels = channels
        self.quality = quality

        self._stream = soxr.ResampleStream(
            input_rate_hz,
            output_rate_hz,
            channels,
            dtype="float32",
            quality=quality,
        )

    def reset(self) -> ResamplerState:
        """Reset the application-level stream state.

        A new libsoxr stream is also created so that no filter history
        survives a reset.
        """
        self._stream = soxr.ResampleStream(
            self.input_rate_hz,
            self.output_rate_hz,
            self.channels,
            dtype="float32",
            quality=self.quality,
        )

        return ResamplerState()

    def process(
        self,
        audio: np.ndarray,
        state: ResamplerState,
    ) -> tuple[np.ndarray, ResamplerState]:
        """Process one streaming audio chunk.

        For mono audio, input and output are 1-D arrays.

        For multi-channel audio, input and output are 2-D arrays with
        shape ``(samples, channels)``.
        """
        audio = np.asarray(audio, dtype=np.float32)

        if self.channels == 1:
            if audio.ndim != 1:
                raise ValueError("mono audio must be a 1-D array")
        else:
            if audio.ndim != 2:
                raise ValueError("multi-channel audio must be a 2-D array")

            if audio.shape[1] != self.channels:
                raise ValueError(
                    f"expected {self.channels} channels, got {audio.shape[1]}"
                )

        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite values")

        output = self._stream.resample_chunk(audio, last=False)

        state.samples_processed += audio.shape[0]

        return np.asarray(output, dtype=np.float32), state

    def flush(
        self,
        state: ResamplerState,
    ) -> tuple[np.ndarray, ResamplerState]:
        """Flush buffered samples at the end of a stream."""
        output = self._stream.resample_chunk(
            np.empty((0,), dtype=np.float32)
            if self.channels == 1
            else np.empty((0, self.channels), dtype=np.float32),
            last=True,
        )

        return np.asarray(output, dtype=np.float32), state

    @property
    def ratio(self) -> float:
        """Output samples per input sample."""
        return self.output_rate_hz / self.input_rate_hz
