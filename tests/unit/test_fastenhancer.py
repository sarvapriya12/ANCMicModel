import numpy as np
import pytest

from highspl.models.base import ModelState
from highspl.models.fastenhancer import (
    FastEnhancerAdapter,
    FastEnhancerConfig,
)


class FakeSTFT:
    def initialize_cache(self, _x):
        return [None, None]

    def __call__(self, x, cache):
        return x, cache

    def inverse(self, x, cache):
        return x, cache


class FakeStreamingBackend:
    """Fake exact-hop backend for adapter buffering tests."""

    def __init__(self, hop_size: int) -> None:
        self.hop_size = hop_size
        self.calls = []
        self.stft = FakeSTFT()

    def reset(self) -> None:
        self.calls.clear()

    def initialize_cache(self, _x):
        return [None, None, None]

    def __call__(self, x, *cache):
        self.calls.append(x.detach().cpu().numpy().copy())

        return x, *cache


def config(hop_size: int = 512) -> FastEnhancerConfig:
    return FastEnhancerConfig(
        model_kwargs={
            "sample_rate": 48_000,
            "hop_size": hop_size,
        }
    )


def test_config_accepts_valid_pytorch_config():
    cfg = config()

    assert cfg.backend == "pytorch"
    assert cfg.model_kwargs["hop_size"] == 512


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        FastEnhancerConfig(
            model_kwargs={
                "sample_rate": 48_000,
                "hop_size": 512,
            },
            backend="invalid",
        )


def test_non_48khz_config_is_rejected():
    with pytest.raises(ValueError, match="48 kHz"):
        FastEnhancerConfig(
            model_kwargs={
                "sample_rate": 16_000,
                "hop_size": 512,
            }
        )


def test_missing_hop_size_is_rejected():
    with pytest.raises(ValueError, match="hop_size"):
        FastEnhancerConfig(
            model_kwargs={
                "sample_rate": 48_000,
            }
        )


def test_reset_returns_model_state():
    adapter = FastEnhancerAdapter(config())

    state = adapter.reset()

    assert isinstance(state, ModelState)
    assert state.backend == []


def test_short_input_waits_for_complete_hop():
    adapter = FastEnhancerAdapter(config(hop_size=8))

    fake = FakeStreamingBackend(8)
    adapter._model = fake

    state = adapter.reset()

    audio = np.ones(4, dtype=np.float32)

    output, state = adapter.process(
        audio,
        sample_rate=48_000,
        state=state,
    )

    assert output.size == 0
    assert len(fake.calls) == 0


def test_exact_hop_is_processed():
    adapter = FastEnhancerAdapter(config(hop_size=8))

    fake = FakeStreamingBackend(8)
    adapter._model = fake

    state = adapter.reset()

    audio = np.arange(8, dtype=np.float32)

    output, state = adapter.process(
        audio,
        sample_rate=48_000,
        state=state,
    )

    assert output.shape == (8,)
    assert len(fake.calls) == 1
    assert np.array_equal(fake.calls[0].reshape(-1), audio)


def test_arbitrary_chunks_are_buffered_into_hops():
    adapter = FastEnhancerAdapter(config(hop_size=8))

    fake = FakeStreamingBackend(8)
    adapter._model = fake

    state = adapter.reset()

    first, state = adapter.process(
        np.arange(3, dtype=np.float32),
        48_000,
        state,
    )

    second, state = adapter.process(
        np.arange(3, 11, dtype=np.float32),
        48_000,
        state,
    )

    assert first.size == 0
    assert second.size == 8
    assert len(fake.calls) == 1

    expected = np.arange(8, dtype=np.float32)

    assert np.array_equal(
        fake.calls[0].reshape(-1),
        expected,
    )


def test_multiple_hops_processed_from_one_chunk():
    adapter = FastEnhancerAdapter(config(hop_size=8))

    fake = FakeStreamingBackend(8)
    adapter._model = fake

    state = adapter.reset()

    audio = np.arange(24, dtype=np.float32)

    output, state = adapter.process(
        audio,
        48_000,
        state,
    )

    assert output.shape == (24,)
    assert len(fake.calls) == 3


def test_wrong_sample_rate_is_rejected():
    adapter = FastEnhancerAdapter(config())

    state = adapter.reset()

    with pytest.raises(ValueError, match="expects"):
        adapter.process(
            np.zeros(512, dtype=np.float32),
            32_000,
            state,
        )


def test_non_mono_audio_is_rejected():
    adapter = FastEnhancerAdapter(config())

    state = adapter.reset()

    with pytest.raises(ValueError, match="mono"):
        adapter.process(
            np.zeros((512, 2), dtype=np.float32),
            48_000,
            state,
        )


def test_non_finite_audio_is_rejected():
    adapter = FastEnhancerAdapter(config())

    state = adapter.reset()

    audio = np.zeros(512, dtype=np.float32)
    audio[10] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        adapter.process(
            audio,
            48_000,
            state,
        )


def test_tensorrt_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError, match="TensorRT"):
        FastEnhancerAdapter(
            FastEnhancerConfig(
                model_kwargs={
                    "sample_rate": 48_000,
                    "hop_size": 512,
                },
                backend="tensorrt",
            )
        )
