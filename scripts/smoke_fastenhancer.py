import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(r"D:\SIH\fastenhancer")
MODEL_DIR = PROJECT_ROOT / "models" / "fastenhancer" / "b"

sys.path.insert(0, str(UPSTREAM_ROOT))

from models.fastenhancer.default.model import Model


def main() -> None:
    config_path = MODEL_DIR / "config.yaml"
    checkpoint_path = MODEL_DIR / "00500.pth"

    cfg = OmegaConf.load(config_path)
    model_kwargs = OmegaConf.to_container(
        cfg.model_kwargs,
        resolve=True,
    )

    print("1. Constructing official FastEnhancer Base...")
    model = Model(**model_kwargs)

    print("2. Loading checkpoint strictly...")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    print("   strict_load=OK")

    print("3. Removing weight reparameterizations...")
    model.remove_weight_reparameterizations()
    model.eval()
    model.flatten_parameters()
    print("   reparameterization_removal=OK")

    print("4. Initializing streaming caches...")
    x = torch.zeros(1, 512)

    cache_stft, cache_istft = model.stft.initialize_cache(x)
    cache_model = model.initialize_cache(x)

    print(f"   model_caches={len(cache_model)}")
    print(f"   stft_cache_shape={tuple(cache_stft.shape)}")
    print(f"   istft_cache_shape={tuple(cache_istft.shape)}")

    print("5. Running one exact 512-sample streaming hop...")

    with torch.no_grad():
        spec_in, cache_stft = model.stft(x, cache_stft)

        spec_out, *cache_model = model.model_forward(
            spec_in,
            *cache_model,
        )

        wav_out, cache_istft = model.stft.inverse(
            spec_out,
            cache_istft,
        )

    print(f"   output_shape={tuple(wav_out.shape)}")
    print(f"   output_dtype={wav_out.dtype}")
    print(f"   output_finite={bool(torch.isfinite(wav_out).all())}")

    print("6. Running second hop to verify state continuity...")

    x2 = torch.randn(1, 512)

    with torch.no_grad():
        spec_in, cache_stft = model.stft(x2, cache_stft)

        spec_out, *cache_model = model.model_forward(
            spec_in,
            *cache_model,
        )

        wav_out2, cache_istft = model.stft.inverse(
            spec_out,
            cache_istft,
        )

    print(f"   second_output_shape={tuple(wav_out2.shape)}")
    print(f"   second_output_finite={bool(torch.isfinite(wav_out2).all())}")

    print()
    print("FASTENHANCER_REAL_MODEL_SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
