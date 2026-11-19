"""Listen to the pipeline on your PC (no hardware needed).

Input is either a synthetic scene directory (``data/generated/eval_v1/<id>``)
or any 2-channel WAV/FLAC at 48 kHz (left = primary mic, right = reference mic).
Writes four files you can play: the raw primary mic, the AI-only baseline, the
selected pipeline, and (for scenes) the clean target.

    uv run python scripts/demo_listen.py data/generated/eval_v1/0012_S2_tank_b2
    uv run python scripts/demo_listen.py my_recording.wav --preset gsc_blend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from highspl.data.mixture import read_scene
from highspl.evaluation.runner import (
    BLOCK,
    SR,
    best_lag,
    build_pipeline,
    calibrate_latency,
    default_preset,
    score,
    stream,
)


def _run(preset: str, mic: np.ndarray) -> np.ndarray:
    pipe = build_pipeline(preset)
    lag = calibrate_latency(pipe)
    out, _, _ = stream(pipe, mic)
    return out[lag : lag + mic.shape[1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="scene directory or 2-channel audio file")
    ap.add_argument("--preset", default=None, help="default: best available")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "demo"))
    args = ap.parse_args()

    src = Path(args.source)
    clean = None
    if src.is_dir():
        scene = read_scene(src)
        mic, clean, name = scene.mic, scene.clean, src.name
        print(f"scene {name}: {scene.spec}")
    else:
        data, sr = sf.read(str(src), dtype="float32", always_2d=True)
        if sr != SR or data.shape[1] != 2:
            sys.exit(f"need a 2-channel {SR} Hz file (got {data.shape[1]} ch, {sr} Hz)")
        mic, name = data.T.copy(), src.stem

    preset = args.preset or default_preset()
    out_dir = Path(args.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "1_mic_primary": mic[0],
        "2_ai_only": _run("bypass", mic),
        f"3_pipeline_{preset}": _run(preset, mic),
    }
    if clean is not None:
        results["4_clean_target"] = clean

    peak = max(float(np.max(np.abs(v))) for v in results.values())
    for k, v in results.items():
        # one shared gain so loudness differences stay audible
        sf.write(str(out_dir / f"{k}.wav"), v / max(peak, 1.0), SR, subtype="PCM_16")
        extra = ""
        if clean is not None and k != "4_clean_target":
            s = score(clean, v)
            extra = "  " + "  ".join(
                f"{m}={s[m]:.2f}" for m in ("si_sdr", "estoi", "pesq") if m in s
            )
        print(f"{k:28s}{extra}")
    print(f"\nwritten to {out_dir}  (block={BLOCK}, sr={SR})")
    _ = best_lag  # re-exported for ad-hoc use


if __name__ == "__main__":
    main()
