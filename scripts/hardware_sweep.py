"""Hardware decision sweep: how do AOP, mic spacing and pod distance change the
quality the system can deliver? Re-renders the SAME 24 real-speech scenes (one
per noise group x SPL band) under each hardware variant and evaluates a
pipeline against AI-only.

    uv run python scripts/hardware_sweep.py --pipeline gsc_blend --workers 10

Variants (see VARIANTS): INMP441 AOP 120 dB vs 130/135 dB parts (with the same
self-noise, and with self-noise raised to keep an 87 dB dynamic range), mic
spacing 0.8-3 cm, and wearer distance 4.5-6.5 cm (pipeline designed for it) or
a mismatch between the designed and the real distance.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SEED = 20260101
N_SCENES = 24  # indices 0..23: every (group x SPL band) once

VARIANTS: dict[str, dict] = {
    "base (AOP120, d1.5, r5.5)": {},
    # microphone overload point
    "AOP130 (same self-noise)": {"hw": {"aop_db": 130.0}},
    "AOP130 (87 dB range)": {"hw": {"aop_db": 130.0, "self_noise_db_spl": 43.0}},
    "AOP135 (same self-noise)": {"hw": {"aop_db": 135.0}},
    "AOP135 (87 dB range)": {"hw": {"aop_db": 135.0, "self_noise_db_spl": 48.0}},
    # capsule spacing (pipeline designed for the same spacing)
    "spacing 0.8 cm": {"geo": {"spacing_m": 0.008}, "design": {"mic_spacing_m": 0.008}},
    "spacing 2.0 cm": {"geo": {"spacing_m": 0.02}, "design": {"mic_spacing_m": 0.02}},
    "spacing 3.0 cm": {"geo": {"spacing_m": 0.03}, "design": {"mic_spacing_m": 0.03}},
    # lips-to-primary distance (pipeline designed for it)
    "lips 4.5 cm": {"geo": {"mouth_m": 0.045}, "design": {"mouth_distance_m": 0.045}},
    "lips 6.5 cm": {"geo": {"mouth_m": 0.065}, "design": {"mouth_distance_m": 0.065}},
    # wearer places the pod differently than the pipeline assumes (5.5 cm)
    "lips 4.0 cm, designed 5.5": {"geo": {"mouth_m": 0.04}},
    "lips 7.0 cm, designed 5.5": {"geo": {"mouth_m": 0.07}},
}

_CACHE: dict = {}


def _work(task):
    from highspl.data.mixture import SpeechPool, render_scene, sample_scene_spec
    from highspl.evaluation.runner import build_pipeline, evaluate_scene

    variant, index, pipelines, lags = task
    v = VARIANTS[variant]
    speech = str(ROOT / "data" / "raw" / "librispeech" / "LibriSpeech" / "test-clean")
    if "pool" not in _CACHE:
        _CACHE["pool"] = SpeechPool(speech)
    spec = sample_scene_spec(
        BASE_SEED,
        index,
        duration_s=8.0,
        tier="S2",
        hardware_overrides=v.get("hw"),
        geometry_overrides=v.get("geo"),
    )
    scene = render_scene(spec, _CACHE["pool"])

    over = [f"pipeline.frontend.{k}={val}" for k, val in v.get("design", {}).items()]
    pipes = {}
    for name in pipelines:
        key = (name, tuple(over))
        if key not in _CACHE:
            _CACHE[key] = build_pipeline(name, over)
        pipes[name] = _CACHE[key]
    rows = evaluate_scene(scene, pipes, lags)
    for r in rows:
        r["variant"] = variant
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pipeline", default="gsc_blend")
    ap.add_argument("--baseline", default="bypass")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="substring filter on variant names")
    ap.add_argument(
        "--out", default=str(ROOT / "artifacts" / "eval" / "hardware_sweep")
    )
    args = ap.parse_args(argv)

    from highspl.evaluation.report import aggregate, md_table, write_rows
    from highspl.evaluation.runner import build_pipeline, calibrate_latency

    names = [args.pipeline, args.baseline]
    lags = {n: calibrate_latency(build_pipeline(n)) for n in names}
    variants = [v for v in VARIANTS if not args.only or args.only in v]
    tasks = [(v, i, names, lags) for v in variants for i in range(N_SCENES)]

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_work, tasks, chunksize=1), 1):
            rows += r
            if k % 12 == 0 or k == len(tasks):
                print(f"[{k}/{len(tasks)}] ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(rows, out / "results.csv")

    metrics = ("estoi", "si_sdr", "pesq", "f0_recall", "mod_corr")
    lines = [
        "# Hardware sweep\n",
        (
            f"pipeline `{args.pipeline}` vs `{args.baseline}` (AI only) and the raw primary mic; "
            f"{N_SCENES} real-speech scenes per variant (one per noise group x SPL band).\n"
        ),
    ]
    lines.append("\n## All SPL bands (mean)\n")
    agg = aggregate(
        [r for r in rows if r["pipeline"] != "raw"], ("variant", "pipeline"), metrics
    )
    lines.append(md_table(agg, ["variant", "pipeline", "n", *metrics]))
    lines.append("\n## High-SPL bands only (120-138 dB at the mic)\n")
    hi = [r for r in rows if r["pipeline"] != "raw" and r["band"] >= 2]
    lines.append(
        md_table(
            aggregate(hi, ("variant", "pipeline"), metrics),
            ["variant", "pipeline", "n", *metrics],
        )
    )
    lines.append("\n## By SPL band, ESTOI (candidate pipeline)\n")
    cand = [r for r in rows if r["pipeline"] == args.pipeline]
    table = []
    for v in variants:
        row = {"variant": v}
        for b in range(4):
            sub = [r["estoi"] for r in cand if r["variant"] == v and r["band"] == b]
            row[f"band{b}"] = sum(sub) / max(len(sub), 1)
        table.append(row)
    lines.append(md_table(table, ["variant", "band0", "band1", "band2", "band3"]))
    lines.append(
        "\nBands: 0 = 90-105, 1 = 105-120, 2 = 120-130, 3 = 130-138 dB SPL at the primary mic.\n"
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
