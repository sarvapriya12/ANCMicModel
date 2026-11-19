"""Export a FastEnhancer checkpoint to the streaming ONNX model the pipeline runs.

    uv run python scripts/export_fastenhancer_b.py                      # stock model
    uv run python scripts/export_fastenhancer_b.py \
        --checkpoint artifacts/finetune/best.pth --output models/fastenhancer/b/ft.onnx

The upstream repo (MIT) is looked up in third_party/fastenhancer first, then
../fastenhancer; override with --upstream-root.
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import onnx
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "fastenhancer" / "b"


def _default_upstream() -> Path:
    for cand in (
        PROJECT_ROOT / "third_party" / "fastenhancer",
        PROJECT_ROOT.parent / "fastenhancer",
    ):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        "upstream fastenhancer repo not found; pass --upstream-root"
    )


def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "00500.pth")
    ap.add_argument("--config", type=Path, default=MODEL_DIR / "config.yaml")
    ap.add_argument(
        "--output", type=Path, default=MODEL_DIR / "fastenhancer_b_streaming.onnx"
    )
    ap.add_argument("--upstream-root", type=Path, default=None)
    return ap.parse_args(argv)


ARGS = _parse() if __name__ == "__main__" else _parse([])
CHECKPOINT, CONFIG, OUTPUT = ARGS.checkpoint, ARGS.config, ARGS.output
UPSTREAM_ROOT = ARGS.upstream_root or _default_upstream()

sys.path.insert(0, str(UPSTREAM_ROOT))

from models.fastenhancer.default.model import ONNXModel


class StreamingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def initialize_cache(self, x):
        return [
            *self.model.stft.initialize_cache(x),
            *self.model.initialize_cache(x),
        ]

    def forward(
        self,
        wav_in,
        cache_in_0,
        cache_in_1,
        cache_in_2,
        cache_in_3,
        cache_in_4,
    ):
        spec_in, cache_out_0 = self.model.stft(
            wav_in,
            cache_in_0,
        )

        spec_out, cache_out_2, cache_out_3, cache_out_4 = self.model(
            spec_in,
            cache_in_2,
            cache_in_3,
            cache_in_4,
        )

        wav_out, cache_out_1 = self.model.stft.inverse(
            spec_out,
            cache_in_1,
        )

        return (
            wav_out,
            cache_out_0,
            cache_out_1,
            cache_out_2,
            cache_out_3,
            cache_out_4,
        )


def main():
    print("Loading config...")
    cfg = OmegaConf.load(CONFIG)
    kwargs = OmegaConf.to_container(
        cfg.model_kwargs,
        resolve=True,
    )

    print("Constructing ONNXModel...")
    model = ONNXModel(**kwargs)

    print("Loading checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    print("Removing weight reparameterizations...")
    model.remove_weight_reparameterizations()
    model.eval()
    model.flatten_parameters()

    wrapper = StreamingWrapper(model)
    wrapper.eval()

    print("Initializing caches...")
    x = torch.zeros(1, 512)

    cache_list = wrapper.initialize_cache(torch.zeros(1))

    print("Cache count:", len(cache_list))

    for i, cache in enumerate(cache_list):
        print(f"cache_{i}: shape={tuple(cache.shape)} dtype={cache.dtype}")

    print("Exporting ONNX...")
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.onnx.export(
        wrapper,
        args=(x, *cache_list),
        f=str(OUTPUT),
        input_names=[
            "wav_in",
            "cache_in_0",
            "cache_in_1",
            "cache_in_2",
            "cache_in_3",
            "cache_in_4",
        ],
        output_names=[
            "wav_out",
            "cache_out_0",
            "cache_out_1",
            "cache_out_2",
            "cache_out_3",
            "cache_out_4",
        ],
        dynamo=True,
        external_data=False,
    )

    print("Checking ONNX graph...")
    onnx_model = onnx.load(str(OUTPUT))
    onnx.checker.check_model(onnx_model)

    print("Saving verified ONNX model...")
    onnx.save(
        onnx_model,
        str(OUTPUT),
    )

    print()
    print("ONNX_EXPORT=PASS")
    print(f"OUTPUT={OUTPUT}")


if __name__ == "__main__":
    main()
