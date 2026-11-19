"""Fine-tune the FastEnhancer-B neural stage on the synthetic battlefield set.

    uv run python scripts/make_train_set.py --preset gsc_agc --n 4000
    uv run python scripts/train.py --data data/generated/train_gsc_agc

Then export and evaluate:

    uv run python scripts/export_fastenhancer_b.py \
        --checkpoint artifacts/finetune/best.pth --output models/fastenhancer/b/ft.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "fastenhancer" / "b"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "finetune"))
    ap.add_argument("--checkpoint", default=str(MODEL_DIR / "00500.pth"))
    ap.add_argument("--config", default=str(MODEL_DIR / "config.yaml"))
    ap.add_argument(
        "--upstream-root", default=str(ROOT / "third_party" / "fastenhancer")
    )
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from highspl.training.trainer import FinetuneConfig, run_finetune

    result = run_finetune(
        FinetuneConfig(
            data_dir=args.data,
            out_dir=args.out,
            checkpoint=args.checkpoint,
            config_yaml=args.config,
            upstream_root=args.upstream_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
        )
    )
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
