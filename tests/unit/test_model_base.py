import numpy as np
import pytest

from highspl.models.base import ModelState, StreamingEnhancer


def test_model_state_stores_backend():
    backend = object()

    state = ModelState(backend=backend)

    assert state.backend is backend


def test_streaming_enhancer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        StreamingEnhancer()


class DummyEnhancer(StreamingEnhancer):
    sample_rate_in = 48000
    sample_rate_out = 48000

    def reset(self) -> ModelState:
        return ModelState(backend={})

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        state: ModelState,
    ) -> tuple[np.ndarray, ModelState]:
        return audio.copy(), state

    def flush(
        self,
        state: ModelState,
    ) -> tuple[np.ndarray, ModelState]:
        return np.empty(0, dtype=np.float32), state


def test_dummy_enhancer_implements_interface():
    enhancer = DummyEnhancer()

    state = enhancer.reset()

    audio = np.ones(160, dtype=np.float32)

    output, state = enhancer.process(
        audio,
        sample_rate=48000,
        state=state,
    )

    assert np.array_equal(output, audio)
    assert isinstance(state, ModelState)

    tail, state = enhancer.flush(state)

    assert tail.dtype == np.float32
    assert tail.size == 0


def test_backend_state_can_preserve_streaming_information():
    enhancer = DummyEnhancer()

    state = enhancer.reset()
    state.backend["frames"] = 10

    _, state = enhancer.process(
        np.ones(80, dtype=np.float32),
        sample_rate=48000,
        state=state,
    )

    assert state.backend["frames"] == 10
