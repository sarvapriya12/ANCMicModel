"""Real-time benchmark for the streaming pipeline (run it ON the Raspberry Pi 5).

    uv run python scripts/benchmark_pi5.py --preset gsc_blend --seconds 60
    uv run python scripts/benchmark_pi5.py --preset gsc_blend --soak-minutes 30

A real-time system fails on its *tail*, not its mean: one 15 ms block among
thousands of 3 ms blocks is an audible dropout. The verdict therefore gates on
the 99.9th percentile against the 10.67 ms block deadline (512 samples @ 48 kHz).

The input is a rendered synthetic battlefield scene (jet noise + speech through
the pod geometry and hardware model), not white noise, so the adaptive filters
and gates run their real code paths.

Run on a Pi with the performance governor and an active cooler for a meaningful
soak result. Results from other machines say nothing about the Pi.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 512
SR = 48_000
DEADLINE_MS = BLOCK / SR * 1000.0


def _read(path: str) -> float | None:
    try:
        return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def sys_status() -> dict:
    return {
        "temp_c": (_read("/sys/class/thermal/thermal_zone0/temp") or float("nan"))
        / 1000.0,
        "cpu_mhz": (
            _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            or float("nan")
        )
        / 1000.0,
    }


def make_stream(seconds: float) -> np.ndarray:
    from highspl.data.mixture import render_scene, sample_scene_spec

    spec = sample_scene_spec(
        1, 12, duration_s=max(seconds, 10.0), tier="S1"
    )  # jet, 105-120 dB band
    return render_scene(spec).mic


def run(preset: str, seconds: float, warmup_s: float) -> dict:
    from highspl.evaluation.runner import build_pipeline

    pipe = build_pipeline(preset)
    mic = make_stream(min(seconds, 20.0))
    n_blocks = int(seconds * SR / BLOCK)
    n_warm = int(warmup_s * SR / BLOCK)
    pipe.reset()

    times = np.empty(n_blocks)
    dsp = np.empty(n_blocks)
    ai = np.empty(n_blocks)
    pos = 0
    start = sys_status()
    for k in range(n_blocks + n_warm):
        if pos + BLOCK > mic.shape[1]:
            pos = 0
        d, x = mic[0, pos : pos + BLOCK], mic[1, pos : pos + BLOCK]
        pos += BLOCK
        t0 = time.perf_counter()
        res = pipe.process(d, x)
        dt = (time.perf_counter() - t0) * 1000.0
        if k >= n_warm:
            times[k - n_warm] = dt
            dsp[k - n_warm] = res.telemetry["time_dsp_ms"]
            ai[k - n_warm] = res.telemetry["time_ai_ms"]
    end = sys_status()

    return {
        "preset": preset,
        "blocks": n_blocks,
        "mean": float(times.mean()),
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "p999": float(np.percentile(times, 99.9)),
        "max": float(times.max()),
        "misses": int((times > DEADLINE_MS).sum()),
        "rtf": float(times.mean() / DEADLINE_MS),
        "dsp_mean": float(dsp.mean()),
        "ai_mean": float(ai.mean()),
        "temp_start": start["temp_c"],
        "temp_end": end["temp_c"],
        "mhz_end": end["cpu_mhz"],
    }


def verdict(r: dict) -> str:
    if r["p999"] < 0.8 * DEADLINE_MS and r["misses"] == 0:
        return "PASS"
    if r["p999"] < DEADLINE_MS:
        return "MARGINAL"
    return "FAIL"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="gsc_blend")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument(
        "--soak-minutes",
        type=float,
        default=0.0,
        help="repeat in 60 s rounds for this long",
    )
    args = ap.parse_args(argv)

    rounds = max(round(args.soak_minutes), 1) if args.soak_minutes else 1
    seconds = 60.0 if args.soak_minutes else args.seconds
    print(
        f"preset={args.preset}  block={BLOCK} ({DEADLINE_MS:.2f} ms)  rounds={rounds} x {seconds:.0f}s"
    )
    worst = None
    for i in range(rounds):
        r = run(args.preset, seconds, args.warmup)
        v = verdict(r)
        print(
            f"[{i + 1}/{rounds}] mean {r['mean']:.2f}  p50 {r['p50']:.2f}  p99 {r['p99']:.2f}  "
            f"p99.9 {r['p999']:.2f}  max {r['max']:.2f} ms | misses {r['misses']} | RTF {r['rtf']:.2f} | "
            f"dsp {r['dsp_mean']:.2f} ai {r['ai_mean']:.2f} | {r['temp_end']:.0f} C {r['mhz_end']:.0f} MHz | {v}"
        )
        if worst is None or r["p999"] > worst["p999"]:
            worst = r
    final = verdict(worst)
    print(
        f"\nVERDICT: {final}  (worst p99.9 = {worst['p999']:.2f} ms vs deadline {DEADLINE_MS:.2f} ms)"
    )
    return 0 if final != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
