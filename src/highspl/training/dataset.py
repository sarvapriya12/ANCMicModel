"""Training pairs for the neural stage: (stage-1 output, aligned clean target).

The enhancer is fine-tuned on **exactly the signal it receives at run time**:
the output of the chosen front-end (+ AGC), produced by the very same
``StreamingPipeline.stage1`` the live system uses. The target is the clean
speech at the primary mic, delayed by the front-end's measured delay and scaled
by the AGC gain actually applied (so level is consistent between input and
target and the loss does not fight the AGC).
"""

from __future__ import annotations

import numpy as np

from ..data.mixture import SceneSpec, SpeechPool, render_scene, sample_scene_spec
from ..data.rir import PodGeometry, mouth_to_mics
from ..evaluation.runner import best_lag, build_pipeline

SR = 48_000
BLOCK = 512
TRAIN_BASE_SEED = 777_000
VAL_BASE_SEED = 888_000

_PIPES: dict[str, object] = {}


def _pipe(preset: str):
    if preset not in _PIPES:
        _PIPES[preset] = build_pipeline(preset)
    return _PIPES[preset]


def _run_stage1(pipe, mic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = mic.shape[1]
    pad = (-n) % BLOCK
    d = np.pad(mic[0], (0, pad)).astype(np.float32)
    x = np.pad(mic[1], (0, pad)).astype(np.float32)
    pipe.reset()
    outs, gains = [], []
    for i in range(0, d.size, BLOCK):
        out, gain, *_ = pipe.stage1(d[i : i + BLOCK], x[i : i + BLOCK])
        outs.append(out)
        gains.append(gain)
    return np.concatenate(outs)[:n], np.concatenate(gains)[:n]


def frontend_delay(preset: str) -> int:
    """Samples by which the front-end delays mouth speech (measured on a probe)."""
    rng = np.random.default_rng(0)
    src = np.convolve(rng.standard_normal(SR * 3), np.ones(4) / 4, mode="same") * 0.05
    f, r = mouth_to_mics(src, PodGeometry(), SR)
    out, _ = _run_stage1(_pipe(preset), np.stack((f, r)).astype(np.float32))
    return best_lag(src, out, 0, 256)


def train_spec(
    index: int, duration_s: float = 4.0, base_seed: int = TRAIN_BASE_SEED
) -> SceneSpec:
    """Scene ``index``: ~20 % parametric (S1), 40 % real (S2), 40 % Lombard real (S3)."""
    tier = ("S1", "S1", "S2", "S2", "S2", "S2", "S3", "S3", "S3", "S3")[index % 10]
    return sample_scene_spec(base_seed, index, duration_s=duration_s, tier=tier)


def build_pair(
    spec: SceneSpec, pool: SpeechPool | None, preset: str, delay: int
) -> tuple[np.ndarray, np.ndarray]:
    scene = render_scene(spec, pool)
    x, gain = _run_stage1(_pipe(preset), scene.mic)
    y = np.concatenate((np.zeros(delay, dtype=np.float32), scene.clean))[: x.size]
    return x.astype(np.float32), (y * gain).astype(np.float32)
