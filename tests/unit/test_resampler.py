import numpy as np
import pytest

from highspl.dsp.resampling import ResamplerState, StatefulResampler


def test_reset_creates_fresh_state():
    resampler = StatefulResampler(
        input_rate_hz=32000,
        output_rate_hz=48000,
    )

    state = resampler.reset()

    assert isinstance(state, ResamplerState)
    assert state.samples_processed == 0


def test_ratio_is_correct():
    up = StatefulResampler(32000, 48000)
    down = StatefulResampler(48000, 32000)

    assert up.ratio == pytest.approx(1.5)
    assert down.ratio == pytest.approx(2.0 / 3.0)


def test_32k_to_48k_changes_sample_rate():
    resampler = StatefulResampler(
        input_rate_hz=32000,
        output_rate_hz=48000,
    )
    state = resampler.reset()

    # 100 ms of a sine wave.
    samples = np.arange(3200, dtype=np.float32)
    audio = np.sin(2.0 * np.pi * 440.0 * samples / 32000).astype(np.float32)

    output, state = resampler.process(audio, state)
    tail, state = resampler.flush(state)
    output = np.concatenate([output, tail])

    assert output.dtype == np.float32
    assert abs(len(output) - 4800) <= 2
    assert state.samples_processed == 3200


def test_48k_to_32k_changes_sample_rate():
    resampler = StatefulResampler(
        input_rate_hz=48000,
        output_rate_hz=32000,
    )
    state = resampler.reset()

    samples = np.arange(4800, dtype=np.float32)
    audio = np.sin(2.0 * np.pi * 440.0 * samples / 48000).astype(np.float32)

    output, state = resampler.process(audio, state)
    tail, state = resampler.flush(state)
    output = np.concatenate([output, tail])

    assert output.dtype == np.float32
    assert abs(len(output) - 3200) <= 2
    assert state.samples_processed == 4800


def test_streaming_chunks_match_single_stream():
    rng = np.random.default_rng(1337)

    audio = rng.normal(0.0, 0.1, 32000).astype(np.float32)

    # One continuous stream.
    full_resampler = StatefulResampler(32000, 48000)
    full_state = full_resampler.reset()

    full_output, full_state = full_resampler.process(
        audio,
        full_state,
    )

    # Same audio split across arbitrary chunks.
    chunked_resampler = StatefulResampler(32000, 48000)
    chunked_state = chunked_resampler.reset()

    outputs = []

    start = 0
    chunk_sizes = [137, 511, 73, 1024, 299, 701]

    chunk_index = 0

    while start < len(audio):
        chunk_size = chunk_sizes[chunk_index % len(chunk_sizes)]
        end = min(start + chunk_size, len(audio))

        output, chunked_state = chunked_resampler.process(
            audio[start:end],
            chunked_state,
        )

        outputs.append(output)

        start = end
        chunk_index += 1

    chunked_output = np.concatenate(outputs)

    assert len(chunked_output) == len(full_output)
    assert np.allclose(
        chunked_output,
        full_output,
        atol=1e-5,
        rtol=1e-5,
    )

    assert chunked_state.samples_processed == len(audio)


def test_constant_signal_remains_continuous_across_chunks():
    resampler = StatefulResampler(32000, 48000)
    state = resampler.reset()

    audio = np.ones(3200, dtype=np.float32)

    outputs = []

    for start in range(0, len(audio), 400):
        output, state = resampler.process(
            audio[start : start + 400],
            state,
        )
        outputs.append(output)

    result = np.concatenate(outputs)

    # Ignore filter startup transient.
    steady_state = result[100:]

    assert np.all(np.isfinite(result))
    assert np.mean(np.abs(steady_state - 1.0)) < 1e-3


def test_non_finite_audio_is_rejected():
    resampler = StatefulResampler(32000, 48000)
    state = resampler.reset()

    audio = np.zeros(100, dtype=np.float32)
    audio[50] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        resampler.process(audio, state)


def test_mono_rejects_2d_input():
    resampler = StatefulResampler(32000, 48000)
    state = resampler.reset()

    audio = np.zeros((100, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="mono"):
        resampler.process(audio, state)


def test_multichannel_shape_is_supported():
    resampler = StatefulResampler(
        input_rate_hz=32000,
        output_rate_hz=48000,
        channels=2,
    )
    state = resampler.reset()

    audio = np.zeros((3200, 2), dtype=np.float32)

    output, state = resampler.process(audio, state)
    tail, state = resampler.flush(state)
    output = np.concatenate([output, tail], axis=0)

    assert output.ndim == 2
    assert output.shape[1] == 2
    assert abs(output.shape[0] - 4800) <= 2


@pytest.mark.parametrize(
    "input_rate, output_rate",
    [
        (0, 48000),
        (32000, 0),
        (-32000, 48000),
    ],
)
def test_invalid_sample_rates_are_rejected(input_rate, output_rate):
    with pytest.raises(ValueError, match="rate"):
        StatefulResampler(input_rate, output_rate)


def test_invalid_channels_are_rejected():
    with pytest.raises(ValueError, match="channels"):
        StatefulResampler(32000, 48000, channels=0)


def test_invalid_quality_is_rejected():
    with pytest.raises(ValueError, match="quality"):
        StatefulResampler(
            32000,
            48000,
            quality="INVALID",
        )
