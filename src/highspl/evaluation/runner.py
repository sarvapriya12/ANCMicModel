"""Run pipelines over synthetic scenes through the *real-time* path and score
them. Used by scripts/evaluate.py and the tests."""

from __future__ import annotations

import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np
import soxr
from omegaconf import OmegaConf
from scipy import signal as sps

from ..config import load_config, project_root, resolve_path
from ..data.mixture import Scene, read_scene, render_scene, sample_scene_spec
from ..pipeline import StreamingPipeline
from .metrics import calculate_si_sdr
from .speech_properties import compute_properties

SR = 48_000
BLOCK = 512
TAIL_BLOCKS = 8  # flush the model's algorithmic delay out of the pipeline
DEADLINE_MS = BLOCK / SR * 1000.0


# ---------------------------------------------------------------- pipelines


def preset_path(name: str) -> Path:
    return project_root() / "configs" / "pipelines" / f"{name}.yaml"


def load_pipeline_config(name: str, overrides: list[str] | None = None):
    """Base config merged with ``configs/pipelines/<name>.yaml`` and optional
    dotlist ``overrides`` (e.g. ``["pipeline.frontend.mic_spacing_m=0.02"]``)."""
    cfg = load_config()
    p = preset_path(name)
    if p.is_file():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(p))
    elif name not in ("B0", "base"):
        raise FileNotFoundError(f"unknown pipeline preset {name!r}: {p} not found")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# Best first. A preset is only usable if its ONNX model file exists on this machine.
PRESET_PREFERENCE = ("ft_gsc_blend_agc12", "gsc_blend_agc12", "bypass")


def default_preset() -> str:
    """The best deployable preset whose model is available (``HIGHSPL_PRESET`` overrides).

    The fine-tuned model is a build artefact (``*.onnx`` is git-ignored), so a
    fresh checkout falls back to the stock model with the same front-end."""
    import os

    forced = os.environ.get("HIGHSPL_PRESET")
    if forced:
        return forced
    for name in PRESET_PREFERENCE:
        cfg = load_pipeline_config(name)
        if resolve_path(cfg.model.onnx_path).is_file():
            return name
    return PRESET_PREFERENCE[-1]


def build_pipeline(name: str, overrides: list[str] | None = None) -> StreamingPipeline:
    return StreamingPipeline(load_pipeline_config(name, overrides))


# --------------------------------------------------------------- streaming


def stream(
    pipe: StreamingPipeline, mic: np.ndarray
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Feed ``mic`` (2, N) block by block (zero tail appended to flush the
    algorithmic delay). Returns (output incl. delay, stats, agc_gain) where
    ``agc_gain`` is the per-sample AGC gain on the stage-1 time axis (None if
    the pipeline has no AGC)."""
    n = mic.shape[1]
    total = int(np.ceil(n / BLOCK)) * BLOCK + TAIL_BLOCKS * BLOCK
    d = np.zeros(total, dtype=np.float32)
    x = np.zeros(total, dtype=np.float32)
    d[:n] = mic[0]
    x[:n] = mic[1]

    pipe.reset()
    out = np.empty(total, dtype=np.float32)
    gain = np.ones(total, dtype=np.float32) if pipe._agc is not None else None
    times = np.empty(total // BLOCK)
    unhealthy = 0
    frozen = 0
    for k in range(total // BLOCK):
        sl = slice(k * BLOCK, (k + 1) * BLOCK)
        t0 = time.perf_counter()
        res = pipe.process(d[sl], x[sl])
        times[k] = (time.perf_counter() - t0) * 1000.0
        out[sl] = res.enhanced
        if gain is not None and pipe.last_agc_gain is not None:
            gain[sl] = pipe.last_agc_gain
        unhealthy += 0 if res.telemetry.get("healthy", True) else 1
        frozen += 1 if res.telemetry.get("adaptation_frozen") else 0

    stats = {
        "block_mean_ms": float(times.mean()),
        "block_p50_ms": float(np.percentile(times, 50)),
        "block_p99_ms": float(np.percentile(times, 99)),
        "block_p999_ms": float(np.percentile(times, 99.9)),
        "rtf": float(times.mean() / DEADLINE_MS),
        "deadline_misses": int((times > DEADLINE_MS).sum()),
        "unhealthy_blocks": unhealthy,
        "adaptation_frozen_frac": frozen / len(times),
    }
    return out, stats, gain


def best_lag(ref: np.ndarray, est: np.ndarray, lo: int, hi: int) -> int:
    """Lag in [lo, hi] maximising sum(est[n+lag] * ref[n])."""
    n = ref.size
    est = est[: n + hi + 1]
    c = sps.fftconvolve(
        est.astype(np.float64), ref[::-1].astype(np.float64), mode="full"
    )
    lags = np.arange(c.size) - (n - 1)
    sel = (lags >= lo) & (lags <= hi)
    return int(lags[sel][np.argmax(c[sel])])


def align(est: np.ndarray, lag: int, n: int) -> np.ndarray:
    seg = est[lag : lag + n]
    if seg.size < n:
        seg = np.pad(seg, (0, n - seg.size))
    return seg


def calibrate_on_quiet_clip(pipe: StreamingPipeline, spec) -> dict | None:
    """Run the pipeline's one-off microphone calibration the way a deployment
    would: a few seconds of the wearer speaking in a quiet place, captured by
    the *same device* (same capsule mismatch, I2S offset and pod geometry)."""
    if not getattr(pipe, "calibration_enabled", False):
        return None
    quiet = dataclasses.replace(
        spec,
        scene_id=spec.scene_id + "_cal",
        seed=spec.seed + 1,
        duration_s=4.0,
        tier="S1",
        noise_spl_db=45.0,
        impulse_class=None,
        impulse_peak_db=None,
        diffuse_db=None,
        wind_db=None,
        babble_db=None,
        speech_spl_db=95.0,
    )
    clip = render_scene(quiet)
    return pipe.calibrate(clip.mic[0], clip.mic[1])


def calibrate_latency(pipe: StreamingPipeline) -> int:
    """Algorithmic latency (samples) of ``pipe`` measured on a quiet scene."""
    spec = sample_scene_spec(1, 0, duration_s=4.0, tier="S1")
    spec = dataclasses.replace(
        spec,
        noise_spl_db=50.0,
        impulse_class=None,
        impulse_peak_db=None,
        diffuse_db=None,
        wind_db=None,
        babble_db=None,
        speech_spl_db=105.0,
    )
    scene = render_scene(spec)
    calibrate_on_quiet_clip(pipe, spec)
    out, _, _ = stream(pipe, scene.mic)
    return best_lag(scene.clean, out, 0, 8192)


# ------------------------------------------------------------------ scoring


def objective_scores(
    ref: np.ndarray, est: np.ndarray, sr: int = SR
) -> dict[str, float]:
    from pesq import pesq as _pesq
    from pystoi import stoi as _stoi

    out: dict[str, float] = {"si_sdr": float(calculate_si_sdr(ref, est))}
    r16 = soxr.resample(ref.astype(np.float64), sr, 16_000)
    e16 = soxr.resample(est.astype(np.float64), sr, 16_000)
    m = min(r16.size, e16.size)
    r16, e16 = r16[:m], e16[:m]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out["estoi"] = float(_stoi(r16, e16, 16_000, extended=True))
        except Exception:  # noqa: BLE001 - silent / degenerate output
            out["estoi"] = 0.0
        try:
            out["pesq"] = float(_pesq(16_000, r16, e16, "wb"))
        except Exception:  # noqa: BLE001 - PESQ finds no utterance in the output
            out["pesq"] = 1.0
    return out


def score(ref: np.ndarray, est: np.ndarray) -> dict[str, float]:
    s = objective_scores(ref, est)
    p = compute_properties(ref, est)
    dev = p.pop("_ltass_dev")
    s.update(p)
    s["fric_abs_db"] = abs(p["fric_db"]) if np.isfinite(p["fric_db"]) else float("nan")
    s["_ltass_dev"] = dev
    return s


def scene_columns(scene: Scene) -> dict:
    m = scene.meta
    return {
        "scene_id": scene.spec.scene_id,
        "tier": scene.spec.tier,
        "group": scene.spec.group,
        "band": scene.spec.band,
        "snr_db": m["snr_db"],
        "noise_spl_db": m["noise_spl_db"],
        "clip_front": m["clip_fraction_front"],
    }


def evaluate_scene(
    scene: Scene,
    pipelines: dict[str, StreamingPipeline],
    lags: dict[str, int],
    include_raw: bool = True,
    refine: int = 48,
) -> list[dict]:
    n = scene.clean.size
    ref = scene.clean
    base = scene_columns(scene)
    rows: list[dict] = []

    if include_raw:
        s = score(ref, scene.mic[0])
        s.pop("_ltass_dev")
        rows.append({**base, "pipeline": "raw", "latency_ms": 0.0, **s})

    for name, pipe in pipelines.items():
        cal = calibrate_on_quiet_clip(pipe, scene.spec)
        out, stats, gain = stream(pipe, scene.mic)
        lag0 = lags[name]
        lag = best_lag(ref, out, max(lag0 - refine, 0), lag0 + refine)
        est = align(out, lag, n)
        # The AGC deliberately rescales the signal over time. Scale the clean
        # target by the very same gain (on the stage-1 time axis) so that level
        # normalisation is not scored as distortion; without an AGC this is a no-op.
        ref_eff = ref
        if gain is not None:
            d1 = pipe.stage1_delay()
            g = gain[d1 : d1 + n]
            if g.size < n:
                g = np.pad(g, (0, n - g.size), mode="edge")
            ref_eff = ref * g
        s = score(ref_eff, est)
        s.pop("_ltass_dev")
        extra = {}
        if cal is not None:
            extra = {"cal_gain": cal["gain"], "cal_delay": cal["delay"]}
        rows.append(
            {
                **base,
                "pipeline": name,
                "latency_ms": lag / SR * 1000.0,
                **stats,
                **extra,
                **s,
            }
        )
    return rows


def evaluate_scene_dir(
    scene_dir: str | Path, names: list[str], lags: dict[str, int]
) -> list[dict]:
    scene = read_scene(scene_dir)
    pipes = {n: _cached_pipeline(n) for n in names}
    return evaluate_scene(scene, pipes, lags)


_PIPE_CACHE: dict[str, StreamingPipeline] = {}


def _cached_pipeline(name: str) -> StreamingPipeline:
    if name not in _PIPE_CACHE:
        _PIPE_CACHE[name] = build_pipeline(name)
    return _PIPE_CACHE[name]
