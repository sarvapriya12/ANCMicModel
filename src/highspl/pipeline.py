"""The single streaming pipeline used by the lab UI, the evaluator and the Pi.

    (primary, reference) block
        -> guard  (sanitise non-finite samples)
        -> frontend (stage-1 DSP, mono out)
        -> neural enhancer (stage 2)
        -> output gain
        -> enhanced mono block

Design rules:
* The audio path never raises. A failing stage is soft-reset and the block is
  replaced by a safe fallback; the failure is reported in ``telemetry``.
* The pipeline is configuration driven (see ``highspl.config``). Presets only
  change config values, never code paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from omegaconf import DictConfig

from .config import load_config, resolve_path
from .dsp.agc import LookaheadLimiter, StreamingAGC
from .dsp.calibration import RearCalibrator, estimate_mouth_path
from .dsp.frontend import FrontEnd, make_frontend
from .models.base import ModelState, StreamingEnhancer

SAMPLE_RATE = 48_000
C_SOUND = 343.0


@dataclass
class BlockResult:
    enhanced: np.ndarray
    primary: np.ndarray
    reference: np.ndarray
    telemetry: dict[str, Any] = field(default_factory=dict)


def frontend_kwargs(cfg: DictConfig, mode: str) -> dict[str, Any]:
    """Translate config sections into front-end constructor arguments."""
    n = cfg.nlms
    if mode == "robust":
        return {
            "filter_length": int(n.filter_length),
            "step_size": float(n.step_size),
            "leakage": float(n.leakage),
            "freeze_ratio_db": float(n.freeze_ratio_db),
            "pld_threshold_db": float(n.pld_threshold_db),
            "min_reference_energy": float(n.min_reference_db),
            "max_reference_energy": float(n.max_reference_db),
        }
    if mode == "vss":
        return {
            "filter_length": int(n.filter_length),
            "mu_max": 0.5,
            "mu_min": 0.001,
            "alpha": 0.99,
            "gamma": 0.01,
        }
    if mode == "nlms":
        return {"filter_length": int(n.filter_length), "step_size": float(n.step_size)}
    if mode in {"diff", "adiff", "gsc"}:
        fe = cfg.pipeline.frontend
        return {
            "mic_spacing_m": float(fe.mic_spacing_m),
            "mouth_distance_m": float(fe.mouth_distance_m),
            "sample_rate": SAMPLE_RATE,
        }
    return {}


def build_enhancer(cfg: DictConfig) -> StreamingEnhancer:
    """Construct the neural stage from config (ONNX Runtime by default)."""
    from .models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig

    m = cfg.model
    rt = cfg.runtime
    if m.family != "fastenhancer":
        raise ValueError(f"unsupported model family: {m.family!r}")

    return FastEnhancerAdapter(
        FastEnhancerConfig(
            model_kwargs={"hop_size": int(m.hop_size), "sample_rate": SAMPLE_RATE},
            checkpoint_path=(
                str(resolve_path(m.checkpoint)) if rt.backend == "pytorch" else None
            ),
            onnx_path=(
                str(resolve_path(m.onnx_path)) if rt.backend == "onnxruntime" else None
            ),
            backend=str(rt.backend),
            device=str(rt.device),
            intra_op_threads=int(rt.num_threads),
        )
    )


class NeuralStage:
    """Fault-contained wrapper around a StreamingEnhancer."""

    def __init__(self, enhancer: StreamingEnhancer) -> None:
        self.enhancer = enhancer
        self.state: ModelState = enhancer.reset()
        self.resets = 0

    def reset(self) -> None:
        self.state = self.enhancer.reset()

    def process(self, block: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (enhanced block, healthy). Output length == input length."""
        out = None
        try:
            out, self.state = self.enhancer.process(block, SAMPLE_RATE, self.state)
        except Exception:  # noqa: BLE001 - audio path must never raise
            out = None
        if out is not None and out.shape == block.shape and np.all(np.isfinite(out)):
            return out, True
        # Soft-reset only this stage and fall back to the unprocessed block.
        self.resets += 1
        self.reset()
        return block.copy(), False


class StreamingPipeline:
    def __init__(
        self,
        cfg: DictConfig | None = None,
        enhancer: StreamingEnhancer | None = None,
        frontend: FrontEnd | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.block_size = int(self.cfg.runtime.block_size)

        self._frontends: dict[str, FrontEnd] = {}
        self._mode = ""
        self.set_mode(str(self.cfg.pipeline.frontend.mode), frontend)

        self.neural = NeuralStage(
            enhancer if enhancer is not None else build_enhancer(self.cfg)
        )
        lv = self.cfg.pipeline.level
        self._out_gain = float(10.0 ** (float(lv.output_gain_db) / 20.0))
        self._sanitize = bool(self.cfg.pipeline.guard.sanitize)
        self._clip_thr = float(self.cfg.pipeline.guard.clip_threshold)
        self._agc = (
            StreamingAGC(
                target_dbfs=float(lv.agc_target_dbfs),
                max_gain_db=float(lv.agc_max_gain_db),
                min_gain_db=float(lv.agc_min_gain_db),
                sample_rate=SAMPLE_RATE,
            )
            if lv.agc_enabled
            else None
        )
        self._limiter = (
            LookaheadLimiter(ceiling_dbfs=float(lv.limiter_ceiling_dbfs))
            if lv.limiter_enabled
            else None
        )
        fe_cfg = self.cfg.pipeline.frontend
        d_m, r1 = float(fe_cfg.mic_spacing_m), float(fe_cfg.mouth_distance_m)
        self._cal = RearCalibrator(r1 / (r1 + d_m), d_m / C_SOUND * SAMPLE_RATE)
        self.calibration_enabled = bool(self.cfg.pipeline.calibration.enabled)
        self.last_agc_gain: np.ndarray | None = None

    # -- configuration --------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str, frontend: FrontEnd | None = None) -> None:
        """Switch the stage-1 front-end. The new front-end starts from a clean
        state so stale history from an earlier selection can't leak in."""
        mode = mode.lower()
        if frontend is None:
            frontend = make_frontend(mode, **frontend_kwargs(self.cfg, mode))
            fe_cfg = self.cfg.pipeline.frontend
            if fe_cfg.blend and mode in ("diff", "adiff", "gsc"):
                from .dsp.beamformer import BlendedFrontEnd

                frontend = BlendedFrontEnd(
                    frontend,
                    lo_dbfs=float(fe_cfg.blend_floor_lo_dbfs),
                    hi_dbfs=float(fe_cfg.blend_floor_hi_dbfs),
                    block=int(self.cfg.runtime.block_size),
                    sample_rate=SAMPLE_RATE,
                )
        else:
            frontend.reset()
        self._frontends[mode] = frontend
        self._mode = mode

    @property
    def frontend(self) -> FrontEnd:
        return self._frontends[self._mode]

    def stage1_delay(self) -> int:
        """Samples by which stage 1 delays the primary signal (calibration
        alignment + front-end alignment). Used by the evaluator to map the AGC
        gain curve back onto the clean target's time axis."""
        fe = self.frontend
        delay = getattr(fe, "delay_samples", None)
        if delay is None:
            delay = getattr(getattr(fe, "inner", None), "delay_samples", 0)
        return int(delay) + int(self._cal.latency)

    def calibrate(self, front: np.ndarray, rear: np.ndarray) -> dict[str, float]:
        """Estimate the mouth path (gain, delay) from clean speech picked up by
        both mics and correct the rear channel to the nominal pod. Meant to be
        run once in a quiet place; a no-op unless ``calibration.enabled``."""
        est = estimate_mouth_path(
            np.asarray(front, np.float64), np.asarray(rear, np.float64), SAMPLE_RATE
        )
        if self.calibration_enabled:
            self._cal.set_from_measurement(est["gain"], est["delay"])
        return est

    def reset(self) -> None:
        """Reset every stateful stage (call on every stream start)."""
        self.frontend.reset()
        self.neural.reset()
        self._cal.reset()
        if self._agc is not None:
            self._agc.reset()
        if self._limiter is not None:
            self._limiter.reset()

    # -- streaming ------------------------------------------------------

    def stage1(
        self, primary: np.ndarray, reference: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """guard -> calibration -> front-end -> AGC for one block.

        Returns ``(stage1_out, gain, d, x, telemetry)`` where ``gain`` is the
        per-sample linear AGC gain actually applied (all ones when AGC is off).
        Shared by ``process`` and by the training-set builder so the neural stage
        is fine-tuned on exactly the signal it receives at run time."""
        telemetry: dict[str, Any] = {"healthy": True}

        d = np.asarray(primary, dtype=np.float32)
        x = np.asarray(reference, dtype=np.float32)
        if d.ndim != 1 or x.ndim != 1 or d.shape != x.shape:
            raise ValueError("primary and reference must be equal-length 1-D blocks")

        if self._sanitize:
            bad = ~(np.isfinite(d) & np.isfinite(x))
            if bad.any():
                telemetry["nonfinite_samples"] = int(bad.sum())
                d = np.nan_to_num(d, nan=0.0, posinf=1.0, neginf=-1.0)
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        telemetry["clip_front"] = float(np.mean(np.abs(d) >= self._clip_thr))
        telemetry["clip_rear"] = float(np.mean(np.abs(x) >= self._clip_thr))
        d, x = self._cal.process(d, x)  # identity unless calibrated

        t0 = time.perf_counter()
        try:
            out = self.frontend.process(d, x)
            if not np.all(np.isfinite(out)):
                raise FloatingPointError("front-end produced non-finite output")
        except Exception:  # noqa: BLE001 - audio path must never raise
            self.frontend.reset()
            out = d.copy()
            telemetry["healthy"] = False
            telemetry["frontend_reset"] = True
        telemetry["time_dsp_ms"] = (time.perf_counter() - t0) * 1000.0

        gain = np.ones(out.shape, dtype=np.float32)
        if self._agc is not None:
            out = self._agc.process(out)
            gain = self._agc.last_gain
            telemetry["agc_gain_db"] = self._agc.gain_db
        return out, gain, d, x, telemetry

    def process(self, primary: np.ndarray, reference: np.ndarray) -> BlockResult:
        stage1, gain, d, x, telemetry = self.stage1(primary, reference)
        self.last_agc_gain = gain if self._agc is not None else None
        t1 = time.perf_counter()

        # Stage 2
        enhanced, ok = self.neural.process(stage1)
        if not ok:
            telemetry["healthy"] = False
            telemetry["neural_reset"] = True
        t2 = time.perf_counter()

        enhanced = enhanced * self._out_gain
        if self._limiter is not None:
            enhanced = self._limiter.process(enhanced)

        telemetry["adaptation_frozen"] = self.frontend.adaptation_frozen()
        telemetry["time_ai_ms"] = (t2 - t1) * 1000.0
        return BlockResult(
            enhanced=enhanced.astype(np.float32, copy=False),
            primary=d,
            reference=x,
            telemetry=telemetry,
        )

    def flush(self) -> np.ndarray:
        """Emit audio still buffered inside the neural stage (end of stream)."""
        out, self.neural.state = self.neural.enhancer.flush(self.neural.state)
        return (out * self._out_gain).astype(np.float32, copy=False)
