import numpy as np

from highspl.types import AudioChunk, Enhancer, StreamProcessor, StreamState


def test_audio_chunk_defaults():
    samples = np.zeros(160, dtype=np.float32)

    chunk = AudioChunk(
        samples=samples,
        sample_rate=32000,
    )

    assert chunk.samples is samples
    assert chunk.sample_rate == 32000
    assert chunk.timestamp_samples == 0
    assert chunk.is_last is False


def test_audio_chunk_stereo_shape():
    samples = np.zeros((160, 2), dtype=np.float32)

    chunk = AudioChunk(
        samples=samples,
        sample_rate=32000,
        timestamp_samples=160,
        is_last=True,
    )

    assert chunk.samples.shape == (160, 2)
    assert chunk.sample_rate == 32000
    assert chunk.timestamp_samples == 160
    assert chunk.is_last is True


def test_stream_state_has_independent_payload():
    state_a = StreamState()
    state_b = StreamState()

    state_a.payload["counter"] = 10

    assert state_a.payload["counter"] == 10
    assert state_b.payload == {}


def test_stream_state_accepts_processor_state():
    state = StreamState(
        payload={
            "filter_buffer": np.zeros(256, dtype=np.float32),
            "energy_ema": 0.5,
            "warmup_samples": 320,
        }
    )

    assert state.payload["filter_buffer"].shape == (256,)
    assert state.payload["energy_ema"] == 0.5
    assert state.payload["warmup_samples"] == 320


class DummyProcessor:
    def reset(self):
        return StreamState()

    def process(self, chunk, state):
        return chunk, state

    def flush(self, state):
        return AudioChunk(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=32000,
            is_last=True,
        ), state


class DummyEnhancer:
    def reset(self):
        return StreamState()

    def process(self, chunk, state):
        return chunk, state

    def flush(self, state):
        return AudioChunk(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=32000,
            is_last=True,
        ), state


def test_stream_processor_protocol_shape():
    processor: StreamProcessor = DummyProcessor()

    state = processor.reset()

    chunk = AudioChunk(
        samples=np.ones(160, dtype=np.float32),
        sample_rate=32000,
    )

    output, state = processor.process(chunk, state)

    assert output.samples.shape == (160,)
    assert output.sample_rate == 32000

    tail, state = processor.flush(state)

    assert tail.is_last is True
    assert tail.samples.size == 0


def test_enhancer_protocol_shape():
    enhancer: Enhancer = DummyEnhancer()

    state = enhancer.reset()

    chunk = AudioChunk(
        samples=np.ones(160, dtype=np.float32),
        sample_rate=32000,
    )

    output, state = enhancer.process(chunk, state)

    assert output.samples.shape == (160,)
    assert output.sample_rate == 32000

    tail, state = enhancer.flush(state)

    assert tail.is_last is True
    assert tail.samples.size == 0
