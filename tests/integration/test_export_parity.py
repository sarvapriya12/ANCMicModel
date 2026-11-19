"""The streaming ONNX export must reproduce the PyTorch streaming model it came from.

Runs for the stock checkpoint and, when present, for the fine-tuned one. Skipped
when the artefacts (git-ignored) or the upstream repo are unavailable.
"""

import numpy as np
import pytest

from highspl.config import project_root
from highspl.models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig

ROOT = project_root()
MODEL_DIR = ROOT / "models" / "fastenhancer" / "b"
UPSTREAM = ROOT / "third_party" / "fastenhancer"

CASES = [
    ("stock", MODEL_DIR / "00500.pth", MODEL_DIR / "fastenhancer_b_streaming.onnx"),
    (
        "finetuned",
        ROOT / "artifacts" / "finetune" / "best.pth",
        MODEL_DIR / "ft_gsc_blend_agc12.onnx",
    ),
]

MODEL_KWARGS = {
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
}


def _run(backend, checkpoint, onnx, audio):
    adapter = FastEnhancerAdapter(
        FastEnhancerConfig(
            model_kwargs=MODEL_KWARGS,
            checkpoint_path=str(checkpoint) if backend == "pytorch" else None,
            onnx_path=str(onnx) if backend == "onnxruntime" else None,
            backend=backend,
            device="cpu",
            upstream_root=str(UPSTREAM),
        )
    )
    state = adapter.reset()
    out, _ = adapter.process(audio, 48_000, state)
    return out


@pytest.mark.parametrize(
    ("name", "checkpoint", "onnx"), CASES, ids=[c[0] for c in CASES]
)
def test_onnx_matches_pytorch_streaming(name, checkpoint, onnx):
    if not (checkpoint.is_file() and onnx.is_file() and UPSTREAM.is_dir()):
        pytest.skip(f"{name}: checkpoint / ONNX / upstream repo not available")
    pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    t = np.arange(48_000) / 48_000
    audio = (
        0.1 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t))
        + 0.05 * rng.standard_normal(t.size)
    ).astype(np.float32)

    audio = audio[: 512 * 90]  # a whole number of hops: nothing stays buffered
    ref = _run("pytorch", checkpoint, onnx, audio)
    got = _run("onnxruntime", checkpoint, onnx, audio)

    assert ref.shape == got.shape == audio.shape
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=2e-4)
