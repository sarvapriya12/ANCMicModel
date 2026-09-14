from pathlib import Path

import numpy as np

from highspl.models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "fastenhancer" / "b" / "00500.pth"
ONNX_MODEL = ROOT / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"

def get_config(backend: str):
    return FastEnhancerConfig(
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
        backend=backend,
        device="cpu",
        upstream_root=r"D:\SIH\fastenhancer",
    )

def test_pytorch_onnx_parity():
    adapter_pt = FastEnhancerAdapter(get_config("pytorch"))
    adapter_ort = FastEnhancerAdapter(get_config("onnxruntime"))
    
    state_pt = adapter_pt.reset()
    state_ort = adapter_ort.reset()
    
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(48_000 * 5).astype(np.float32)
    
    out_pt, _ = adapter_pt.process(audio, 48_000, state_pt)
    out_ort, _ = adapter_ort.process(audio, 48_000, state_ort)
    
    diff = np.abs(out_pt - out_ort)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"Max absolute error: {max_diff}")
    print(f"Mean absolute error: {mean_diff}")
    
    # Asserting standard floating point tolerance
    np.testing.assert_allclose(out_pt, out_ort, rtol=1e-3, atol=5e-4)

