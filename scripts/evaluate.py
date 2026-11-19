"""Evaluate pipeline presets on a generated scene set.

    uv run python scripts/generate_mixtures.py --set eval_v1 --workers 8
    uv run python scripts/evaluate.py --set eval_v1 --pipelines B0,bypass,fdaf

Every pipeline runs through the same block-by-block streaming path as live
audio. Results: <out>/results.csv and <out>/report.md (with the non-regression
gate of every candidate against ``--baseline``).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _work(scene_dir: str, names: list[str], lags: dict[str, int]):
    from highspl.evaluation.runner import evaluate_scene_dir

    return evaluate_scene_dir(scene_dir, names, lags)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="eval_v1")
    ap.add_argument("--data", default=str(ROOT / "data" / "generated"))
    ap.add_argument("--pipelines", default="B0", help="comma separated preset names")
    ap.add_argument(
        "--baseline", default=None, help="gate every candidate against this"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--tier", default=None, help="only these tiers, comma separated (S1,S2,S3)"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from highspl.evaluation.report import write_report
    from highspl.evaluation.runner import build_pipeline, calibrate_latency

    names = [n.strip() for n in args.pipelines.split(",") if n.strip()]
    scene_dirs = sorted(p for p in (Path(args.data) / args.set).iterdir() if p.is_dir())
    if args.tier:
        tiers = [t.strip() for t in args.tier.split(",")]
        scene_dirs = [d for d in scene_dirs if any(f"_{t}_" in d.name for t in tiers)]
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    if not scene_dirs:
        print(
            "no scenes found; run scripts/generate_mixtures.py first", file=sys.stderr
        )
        return 2

    print(f"calibrating latency for {names} ...", flush=True)
    lags = {n: calibrate_latency(build_pipeline(n)) for n in names}
    for n, lag in lags.items():
        print(f"  {n:12s} algorithmic latency {lag} samples = {lag / 48.0:.1f} ms")

    t0 = time.time()
    rows: list[dict] = []
    fn = partial(_work, names=names, lags=lags)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(fn, [str(d) for d in scene_dirs]), 1):
            rows += r
            print(
                f"[{k:3d}/{len(scene_dirs)}] {r[0]['scene_id']}  ({time.time() - t0:5.0f}s)",
                flush=True,
            )

    out_dir = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "eval" / f"{args.set}_{'-'.join(names)}"
    )
    report = write_report(rows, out_dir, baseline=args.baseline)
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
