"""Scene composer: speech + battlefield noise -> what both mics deliver.

A *scene spec* (plain JSON-able dict) fully determines a scene together with
its seed, so the frozen evaluation set is just a list of specs + a hash.

Level model (all at the front/primary mic, in dB SPL):
  * noise_spl      rms of the stationary noise mix
  * impulse peak   gunfire / artillery peak SPL
  * speech_spl     active-speech rms. Lombard rule: talker level at 1 m is
                   55 + 0.6*max(noise_dBA - 50, 0) dB, capped at 92 dB, plus the
                   near-field gain 20*log10(1 m / r1) (~+25 dB at 5.5 cm)

The clean target is the speech at the primary mic passed through only the
*linear* parts of the hardware chain: exactly what a perfect enhancer outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr

from .clipping import (
    MicHardware,
    apply_chain,
    apply_linear_chain,
    pa_rms_to_spl,
    spl_to_pa_rms,
)
from .noise_synth import (
    CONTINUOUS_CLASSES,
    IMPULSIVE,
    babble,
    make_noise,
)
from .rir import (
    PodGeometry,
    diffuse_pair,
    high_shelf,
    mouth_to_mics,
    plane_wave_to_mics,
)
from .speech_synth import active_speech_rms, synthesize_speech

SR = 48_000

# Stratification used by the frozen evaluation set.
GROUPS = ("jet", "helicopter", "tank", "missile", "gunfire", "artillery")
SPL_BANDS = ((90.0, 105.0), (105.0, 120.0), (120.0, 130.0), (130.0, 138.0))
TIERS = ("S1", "S2", "S3")


# ------------------------------------------------------------ speech pool


class SpeechPool:
    """Real clean speech (LibriSpeech FLACs, 16 kHz -> 48 kHz)."""

    def __init__(self, root: str | Path, sr: int = SR) -> None:
        self.root = Path(root)
        self.sr = sr
        self.files = sorted(self.root.rglob("*.flac"))
        if not self.files:
            raise FileNotFoundError(f"no .flac speech found under {self.root}")

    def speaker_of(self, path: Path) -> str:
        return path.parent.parent.name

    def get(
        self, duration_s: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        """Concatenate utterances of ONE speaker until ``duration_s`` is filled."""
        first = self.files[int(rng.integers(0, len(self.files)))]
        speaker = self.speaker_of(first)
        mine = [f for f in self.files if self.speaker_of(f) == speaker]
        out: list[np.ndarray] = []
        used: list[str] = []
        total = 0
        need = int(duration_s * self.sr)
        while total < need:
            f = mine[int(rng.integers(0, len(mine)))]
            x, fs = sf.read(str(f), dtype="float64")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if fs != self.sr:
                x = soxr.resample(x, fs, self.sr)
            gap = np.zeros(int(rng.uniform(0.1, 0.4) * self.sr))
            out += [x, gap]
            used.append(f.stem)
            total += x.size + gap.size
        wave = np.concatenate(out)
        start = int(rng.integers(0, max(wave.size - need, 0) + 1))
        return wave[start : start + need], used


# --------------------------------------------------------------- the spec


@dataclass
class SceneSpec:
    scene_id: str
    seed: int
    duration_s: float
    tier: str
    group: str
    band: int
    noise_classes: list[str]
    noise_offsets_db: list[float]
    noise_theta_deg: list[float]
    noise_spl_db: float
    impulse_class: str | None
    impulse_peak_db: float | None
    impulse_theta_deg: float
    diffuse_db: float | None
    wind_db: float | None
    babble_db: float | None
    speech_spl_db: float
    lombard: float
    geometry: dict[str, float]
    hardware: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> SceneSpec:
        d = dict(d)
        d["hardware"] = dict(d["hardware"])
        d["hardware"]["mismatch_db"] = tuple(d["hardware"]["mismatch_db"])
        return SceneSpec(**d)


def lombard_speech_spl(
    noise_spl_db: float, mouth_m: float, rng: np.random.Generator
) -> float:
    """Active-speech SPL at the primary mic for a talker raising their voice
    against ``noise_spl_db`` (Lombard effect)."""
    noise_dba = noise_spl_db - 5.0
    at_1m = float(np.clip(55.0 + 0.6 * max(noise_dba - 50.0, 0.0), 55.0, 92.0))
    at_1m += rng.uniform(-3.0, 3.0)
    return at_1m + 20.0 * np.log10(1.0 / mouth_m)


def sample_scene_spec(
    base_seed: int,
    index: int,
    duration_s: float = 10.0,
    tier: str | None = None,
    group: str | None = None,
    band: int | None = None,
    hardware_overrides: dict[str, Any] | None = None,
    geometry_overrides: dict[str, float] | None = None,
) -> SceneSpec:
    """Deterministically draw scene ``index``. ``group``/``band``/``tier`` are
    stratified by index unless given."""
    rng = np.random.default_rng([base_seed, index])
    group = group or GROUPS[index % len(GROUPS)]
    band = band if band is not None else (index // len(GROUPS)) % len(SPL_BANDS)
    tier = tier or TIERS[(index // (len(GROUPS) * len(SPL_BANDS))) % len(TIERS)]
    lo, hi = SPL_BANDS[band]
    noise_spl = float(rng.uniform(lo, hi))

    impulse_class = None
    impulse_peak = None
    if group in ("gunfire", "artillery"):
        impulse_class = group
        # continuous background a little below the stratum level
        noise_spl -= float(rng.uniform(0.0, 10.0))
        impulse_peak = float(min(noise_spl + rng.uniform(12.0, 30.0), 170.0))
        n_cont = 1
        cont = [str(rng.choice(CONTINUOUS_CLASSES))]
    else:
        cont = [group]
        n_cont = int(rng.integers(1, 3))
        while len(cont) < n_cont:
            c = str(rng.choice(CONTINUOUS_CLASSES))
            if c not in cont:
                cont.append(c)
    offsets = [0.0] + [-float(rng.uniform(3.0, 12.0)) for _ in range(len(cont) - 1)]
    thetas = [float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0)))) for _ in cont]

    geom = {
        "spacing_m": float(rng.uniform(0.01, 0.02)),
        "mouth_m": float(rng.uniform(0.045, 0.065)),
        "shadow_db": float(rng.uniform(1.0, 5.0)),
    }
    geom.update(geometry_overrides or {})

    hw: dict[str, Any] = {
        "aop_db": 120.0,
        "self_noise_db_spl": 33.0,
        "mismatch_db": (float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))),
        "i2s_offset_samples": float(rng.uniform(0.0, 1.0)),
        # ALSA overruns are rare on a healthy Pi: most scenes have none.
        "dropout_prob": float(rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.002])),
        "dropout_period_samples": 512,
    }
    hw.update(hardware_overrides or {})

    lombard = float(np.clip((noise_spl - 70.0) / 50.0, 0.0, 1.0))
    return SceneSpec(
        scene_id=f"{index:04d}_{tier}_{group}_b{band}",
        seed=int(rng.integers(0, 2**31 - 1)),
        duration_s=duration_s,
        tier=tier,
        group=group,
        band=band,
        noise_classes=cont,
        noise_offsets_db=offsets,
        noise_theta_deg=thetas,
        noise_spl_db=noise_spl,
        impulse_class=impulse_class,
        impulse_peak_db=impulse_peak,
        impulse_theta_deg=float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0)))),
        diffuse_db=float(-rng.uniform(6.0, 15.0)) if rng.random() < 0.5 else None,
        wind_db=float(-rng.uniform(0.0, 12.0)) if rng.random() < 0.3 else None,
        babble_db=float(rng.uniform(70.0, 90.0)) if rng.random() < 0.15 else None,
        speech_spl_db=float(lombard_speech_spl(noise_spl, geom["mouth_m"], rng)),
        lombard=lombard,
        geometry=geom,
        hardware=hw,
    )


# ------------------------------------------------------------- rendering


@dataclass
class Scene:
    mic: np.ndarray  # (2, N) float32: [primary/front, reference/rear]
    clean: np.ndarray  # (N,) float32 target
    spec: SceneSpec
    meta: dict[str, Any]
    labels: dict[str, Any] = field(default_factory=dict)


def lombard_transform(
    x: np.ndarray, sr: int, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Approximate Lombard effect on real speech: raise pitch/formants a little
    (resampling) and flatten the spectral tilt (HF shelf). Output is longer than
    needed by the caller's crop; its length shrinks by the resampling ratio."""
    if strength <= 0:
        return x
    factor = 1.0 + 0.12 * strength * rng.uniform(0.5, 1.0)
    y = soxr.resample(x, round(sr * factor), sr)  # reads x as faster-> higher pitch
    return high_shelf(y, +6.0 * strength, sr, f0=1500.0)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, max(n - x.size, 0)))[:n]


def render_scene(spec: SceneSpec, pool: SpeechPool | None = None) -> Scene:
    sr = SR
    rng = np.random.default_rng(spec.seed)
    n = round(spec.duration_s * sr)
    geom = PodGeometry(**spec.geometry)
    hw_d = dict(spec.hardware)
    hw = MicHardware(**hw_d)

    # --- speech at the primary mic (pascals) ---------------------------------
    labels: dict[str, Any] = {}
    if spec.tier == "S1":
        clip = synthesize_speech(spec.duration_s, rng, sr=sr, lombard=spec.lombard)
        speech = _fit(clip.wave.astype(np.float64), n)
        labels = clip.labels
    else:
        if pool is None:
            raise ValueError(f"tier {spec.tier} needs a SpeechPool (real speech)")
        extra = 1.25 if spec.tier == "S3" else 1.0
        raw, used = pool.get(spec.duration_s * extra, rng)
        if spec.tier == "S3":
            raw = lombard_transform(raw, sr, spec.lombard, rng)
        speech = _fit(raw, n)
        labels = {"utterances": used}
    speech = speech / (active_speech_rms(speech, sr) + 1e-12)
    speech_pa = speech * spl_to_pa_rms(spec.speech_spl_db)

    # --- stationary noise ------------------------------------------------------
    front_s = np.zeros(n)
    rear_s = np.zeros(n)

    def add(fr: np.ndarray, rr: np.ndarray, gain: float = 1.0) -> None:
        nonlocal front_s, rear_s
        front_s += gain * _fit(fr, n)
        rear_s += gain * _fit(rr, n)

    for cls, off, theta in zip(
        spec.noise_classes, spec.noise_offsets_db, spec.noise_theta_deg
    ):
        src = make_noise(cls, n, sr, rng)
        fr, rr = plane_wave_to_mics(src.signal, theta, geom, sr)
        add(fr, rr, 10.0 ** (off / 20.0))

    base_rms = float(np.sqrt(np.mean(front_s**2)))
    if spec.diffuse_db is not None:
        c0 = spec.noise_classes[0]
        a = make_noise(c0, n, sr, rng).signal
        b = make_noise(c0, n, sr, rng).signal
        fr, rr = diffuse_pair(a, b, geom.spacing_m, sr)
        add(fr, rr, base_rms * 10.0 ** (spec.diffuse_db / 20.0))
    if spec.wind_db is not None:
        w_f = make_noise("wind", n, sr, rng).signal
        w_r = make_noise("wind", n, sr, rng).signal  # independent per capsule
        front_s += base_rms * 10.0 ** (spec.wind_db / 20.0) * _fit(w_f, n)
        rear_s += base_rms * 10.0 ** (spec.wind_db / 20.0) * _fit(w_r, n)

    k = spl_to_pa_rms(spec.noise_spl_db) / (float(np.sqrt(np.mean(front_s**2))) + 1e-30)
    front_s *= k
    rear_s *= k

    front = front_s.copy()
    rear = rear_s.copy()

    # --- far-field babble (absolute level) ------------------------------------
    if spec.babble_db is not None:
        clips = [synthesize_speech(spec.duration_s, rng, sr=sr).wave for _ in range(3)]
        bab = babble(clips, n, rng).signal * spl_to_pa_rms(spec.babble_db)
        theta = float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0))))
        fr, rr = plane_wave_to_mics(bab, theta, geom, sr)
        front += _fit(fr, n)
        rear += _fit(rr, n)

    # --- impulses (peak level at the front mic) -----------------------------------
    if spec.impulse_class is not None:
        src = make_noise(spec.impulse_class, n, sr, rng)
        assert src.kind == IMPULSIVE
        fr, rr = plane_wave_to_mics(src.signal, spec.impulse_theta_deg, geom, sr)
        peak = float(np.max(np.abs(fr))) + 1e-30
        g = spl_to_pa_rms(spec.impulse_peak_db) / peak
        front += g * _fit(fr, n)
        rear += g * _fit(rr, n)

    noise_only_front = front.copy()

    # --- speech through the near-field pod geometry -----------------------------
    sp_f, sp_r = mouth_to_mics(speech_pa, geom, sr)
    front = front + sp_f
    rear = rear + sp_r

    # --- hardware chain ---------------------------------------------------------------
    mic_f, mic_r, info = apply_chain(front, rear, hw, sr, rng)
    clean = apply_linear_chain(speech_pa, hw, sr, mic="front")

    noise_rms = float(np.sqrt(np.mean(noise_only_front**2)))
    meta = {
        "scene_id": spec.scene_id,
        "tier": spec.tier,
        "group": spec.group,
        "band": spec.band,
        "noise_spl_db": pa_rms_to_spl(float(np.sqrt(np.mean(front_s**2)))),
        "total_noise_rms_spl_db": pa_rms_to_spl(noise_rms),
        "speech_spl_db": spec.speech_spl_db,
        "snr_db": spec.speech_spl_db - pa_rms_to_spl(noise_rms),
        "clip_fraction_front": info["clip_fraction_front"],
        "clip_fraction_rear": info["clip_fraction_rear"],
        "lost_fraction": info["lost_fraction"],
        "peak_pa_front": float(np.max(np.abs(front))),
    }
    return Scene(
        mic=np.stack([mic_f, mic_r]).astype(np.float32),
        clean=clean.astype(np.float32),
        spec=spec,
        meta=meta,
        labels=labels,
    )


# ---------------------------------------------------------------------- I/O


def write_scene(scene: Scene, out_dir: str | Path) -> dict[str, str]:
    """Write ``mic.flac`` (24-bit stereo), ``clean.flac`` (24-bit) and
    ``scene.json``; return sha256 hashes of the audio files."""
    out = Path(out_dir) / scene.spec.scene_id
    out.mkdir(parents=True, exist_ok=True)
    sf.write(str(out / "mic.flac"), scene.mic.T, SR, subtype="PCM_24", format="FLAC")
    sf.write(str(out / "clean.flac"), scene.clean, SR, subtype="PCM_24", format="FLAC")

    labels = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in scene.labels.items()
    }
    (out / "scene.json").write_text(
        json.dumps(
            {"spec": scene.spec.to_json(), "meta": scene.meta, "labels": labels},
            indent=1,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
    )
    return {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("mic.flac", "clean.flac")
    }


def read_scene(scene_dir: str | Path) -> Scene:
    d = Path(scene_dir)
    mic, sr = sf.read(str(d / "mic.flac"), dtype="float32")
    clean, _ = sf.read(str(d / "clean.flac"), dtype="float32")
    assert sr == SR
    doc = json.loads((d / "scene.json").read_text())
    labels = {
        k: (np.asarray(v) if isinstance(v, list) and k not in ("phones",) else v)
        for k, v in doc["labels"].items()
    }
    return Scene(
        mic=mic.T.copy(),
        clean=clean,
        spec=SceneSpec.from_json(doc["spec"]),
        meta=doc["meta"],
        labels=labels,
    )
