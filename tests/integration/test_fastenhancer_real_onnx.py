from pathlib import Path

import numpy as np

from highspl.models.fastenhancer import (
    FastEnhancerAdapter,
    FastEnhancerConfig,
)

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = ROOT / "models" / "fastenhancer" / "b" / "00500.pth"

ONNX_MODEL = ROOT / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"


def make_adapter() -> FastEnhancerAdapter:
    return FastEnhancerAdapter(
        FastEnhancerConfig(
            model_kwargs={
                "channels": 48,
                "kernel_size": [8, 3, 3],
                "stride": 4,
                "rnnformer_kwargs": {
                    "num_blocks": 3,
                    "channels": 36,
                    "freq": 36,
                    "num_heads": 4,
                    "eps": 1.0e-5,
                    "positional_embedding": "train",
                    "attn_bias": False,
                    "post_act": False,
                    "pre_norm": False,
                },
                "pre_post_init": "linear",
                "n_fft": 1024,
                "hop_size": 512,
                "win_size": 1024,
                "window": "hann",
                "stft_normalized": False,
                "mask": None,
                "activation": "SiLU",
                "activation_kwargs": {"inplace": True},
                "input_compression": 0.3,
                "normalize_final_conv": True,
                "weight_norm": True,
                "resnet": False,
                "sample_rate": 48_000,
            },
            checkpoint_path=str(CHECKPOINT),
            onnx_path=str(ONNX_MODEL),
            backend="onnxruntime",
            device="cpu",
            upstream_root=r"D:\SIH\fastenhancer",
        )
    )


def test_real_model_single_hop():
    adapter = make_adapter()
    state = adapter.reset()

    audio = np.random.default_rng(0).standard_normal(512).astype(np.float32)

    output, state = adapter.process(
        audio,
        48_000,
        state,
    )

    assert output.shape == (512,)
    assert np.isfinite(output).all()


def test_real_model_multiple_hops():
    adapter = make_adapter()
    state = adapter.reset()

    audio = np.random.default_rng(1).standard_normal(512 * 10).astype(np.float32)

    output, state = adapter.process(
        audio,
        48_000,
        state,
    )

    assert output.shape == audio.shape
    assert np.isfinite(output).all()


def test_real_model_arbitrary_chunks():
    rng = np.random.default_rng(2)
    audio = rng.standard_normal(512 * 8 + 173).astype(np.float32)

    adapter = make_adapter()
    state = adapter.reset()

    outputs = []

    chunk_sizes = [37, 100, 913, 251, 64, 700, 19]

    offset = 0
    index = 0

    while offset < audio.size:
        size = chunk_sizes[index % len(chunk_sizes)]
        chunk = audio[offset : offset + size]

        output, state = adapter.process(
            chunk,
            48_000,
            state,
        )

        if output.size:
            outputs.append(output)

        offset += chunk.size
        index += 1

    flushed, state = adapter.flush(state)

    if flushed.size:
        outputs.append(flushed)

    result = np.concatenate(outputs)

    assert result.shape == audio.shape
    assert np.isfinite(result).all()


def test_real_model_reset_is_deterministic():
    rng = np.random.default_rng(3)
    audio = rng.standard_normal(512 * 4).astype(np.float32)

    adapter = make_adapter()

    state = adapter.reset()
    output_a, _ = adapter.process(
        audio,
        48_000,
        state,
    )

    state = adapter.reset()
    output_b, _ = adapter.process(
        audio,
        48_000,
        state,
    )

    np.testing.assert_allclose(
        output_a,
        output_b,
        rtol=1e-5,
        atol=1e-6,
    )


def test_real_model_silence_is_finite():
    adapter = make_adapter()
    state = adapter.reset()

    audio = np.zeros(512 * 10, dtype=np.float32)

    output, _ = adapter.process(
        audio,
        48_000,
        state,
    )

    assert output.shape == audio.shape
    assert np.isfinite(output).all()


def test_real_model_input_is_not_modified():
    adapter = make_adapter()
    state = adapter.reset()

    audio = np.random.default_rng(4).standard_normal(512).astype(np.float32)
    original = audio.copy()

    adapter.process(
        audio,
        48_000,
        state,
    )

    np.testing.assert_array_equal(audio, original)


def test_real_model_long_stream_is_stable():
    from pathlib import Path

    import numpy as np

    from highspl.models.fastenhancer import (
        FastEnhancerAdapter,
        FastEnhancerConfig,
    )

    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "models" / "fastenhancer" / "b" / "00500.pth"
    onnx_model = (
        root / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"
    )

    adapter = FastEnhancerAdapter(
        FastEnhancerConfig(
            model_kwargs={
                "channels": 48,
                "kernel_size": [8, 3, 3],
                "stride": 4,
                "rnnformer_kwargs": {
                    "num_blocks": 3,
                    "channels": 36,
                    "freq": 36,
                    "num_heads": 4,
                    "eps": 1.0e-5,
                    "positional_embedding": "train",
                    "attn_bias": False,
                    "post_act": False,
                    "pre_norm": False,
                },
                "pre_post_init": "linear",
                "n_fft": 1024,
                "hop_size": 512,
                "win_size": 1024,
                "window": "hann",
                "stft_normalized": False,
                "mask": None,
                "activation": "SiLU",
                "activation_kwargs": {"inplace": True},
                "input_compression": 0.3,
                "normalize_final_conv": True,
                "weight_norm": True,
                "resnet": False,
                "sample_rate": 48_000,
            },
            checkpoint_path=str(checkpoint),
            onnx_path=str(onnx_model),
            backend="onnxruntime",
            device="cpu",
            upstream_root=r"D:\SIH\fastenhancer",
        )
    )

    state = adapter.reset()

    rng = np.random.default_rng(123)

    # ~60 seconds at 48 kHz.
    audio = rng.standard_normal(48_000 * 60).astype(np.float32)

    outputs = []

    # Deliberately irregular chunk sizes.
    chunk_sizes = [97, 512, 333, 1000, 271, 2048, 73, 619]

    offset = 0
    index = 0

    while offset < audio.size:
        size = chunk_sizes[index % len(chunk_sizes)]
        chunk = audio[offset : offset + size]

        output, state = adapter.process(
            chunk,
            48_000,
            state,
        )

        if output.size:
            outputs.append(output)

        assert np.isfinite(output).all()

        offset += chunk.size
        index += 1

    flushed, state = adapter.flush(state)

    if flushed.size:
        outputs.append(flushed)

    result = np.concatenate(outputs)

    assert result.shape == audio.shape
    assert np.isfinite(result).all()
