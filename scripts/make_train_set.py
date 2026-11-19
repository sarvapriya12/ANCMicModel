"""Render the fine-tuning set: stage-1 outputs and aligned clean targets.

    uv run python scripts/make_train_set.py --preset gsc_agc --n 4000 --workers 10

Writes data/generated/train_<preset>/{x,y}.npy (float16, N x 192000) and the
validation split val_x/val_y.npy (different seeds). Speech comes from LibriSpeech
dev-clean; the evaluation set uses test-clean, so speakers are disjoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG = 4.0


def _job(args):
    from highspl.data.mixture import SpeechPool
    from highspl.training.dataset import (
        TRAIN_BASE_SEED,
        VAL_BASE_SEED,
        build_pair,
        train_spec,
    )

    index, preset, delay, split, speech_dir = args
    pool = _pool(speech_dir, SpeechPool)
    base = TRAIN_BASE_SEED if split == "train" else VAL_BASE_SEED
    spec = train_spec(index, SEG, base_seed=base)
    x, y = build_pair(spec, pool if spec.tier != "S1" else None, preset, delay)
    return index, x.astype(np.float16), y.astype(np.float16)


_POOL = {}


def _pool(speech_dir, cls):
    if speech_dir not in _POOL:
        _POOL[speech_dir] = cls(speech_dir)
    return _POOL[speech_dir]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="gsc_agc")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=240)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from highspl.training.dataset import frontend_delay

    speech_dir = str(
        ROOT / "data" / "raw" / "librispeech" / "LibriSpeech" / "dev-clean"
    )
    if not Path(speech_dir).is_dir():
        print("missing dev-clean; run scripts/download_data.py", file=sys.stderr)
        return 2
    delay = frontend_delay(args.preset)
    print(f"front-end '{args.preset}' delay: {delay} samples", flush=True)

    out = (
        Path(args.out)
        if args.out
        else ROOT / "data" / "generated" / f"train_{args.preset}"
    )
    out.mkdir(parents=True, exist_ok=True)
    length = int(SEG * 48_000)

    for split, n, prefix in (("train", args.n, ""), ("val", args.n_val, "val_")):
        xs = np.lib.format.open_memmap(
            out / f"{prefix}x.npy", "w+", np.float16, (n, length)
        )
        ys = np.lib.format.open_memmap(
            out / f"{prefix}y.npy", "w+", np.float16, (n, length)
        )
        jobs = [(i, args.preset, delay, split, speech_dir) for i in range(n)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for k, (i, x, y) in enumerate(ex.map(_job, jobs, chunksize=4), 1):
                xs[i], ys[i] = x, y
                if k % 100 == 0 or k == n:
                    print(f"[{split}] {k}/{n}  ({time.time() - t0:5.0f}s)", flush=True)
        xs.flush()
        ys.flush()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
