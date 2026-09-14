from typing import Any

from highspl.models.base import StreamingEnhancer
from highspl.models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig


def register_models(registry: Any) -> None:
    """Register FastEnhancer models with the model registry."""
    
    def create_fastenhancer_base(kwargs: dict[str, Any]) -> StreamingEnhancer:
        # Default kwargs based on the integration test
        model_kwargs = {
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
        
        if "model_kwargs" in kwargs:
            model_kwargs.update(kwargs["model_kwargs"])
        
        config = FastEnhancerConfig(
            model_kwargs=model_kwargs,
            checkpoint_path=kwargs.get("checkpoint_path"),
            onnx_path=kwargs.get("onnx_path"),
            backend=kwargs.get("backend", "onnxruntime"),
            device=kwargs.get("device", "cpu"),
            upstream_root=kwargs.get("upstream_root"),
        )
        return FastEnhancerAdapter(config)

    registry.register("fastenhancer_b", create_fastenhancer_base)
