"""Parametric source-filter voice (speech tier S1) with exact ground truth.

A glottal pulse train with a time-varying F0 contour (declination, accents,
jitter, shimmer) excites cascaded 2-pole formant resonators whose centre
frequencies move between Peterson-Barney vowel targets. Fricatives (/s/, /sh/,
/f/) and plosives (/t/, /k/, /p/: closure + burst + aspiration) are shaped noise.
A syllable/word/phrase planner produces realistic timing, pauses (20-40 %) and
a 3-6 Hz syllabic modulation.

The generator also emits the labels the evaluation needs: F0 track, voiced
mask, speech-activity mask, per-frame formants and the phone segmentation.
Frame labels use a 10 ms hop (480 samples at 48 kHz).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps
from scipy.interpolate import CubicSpline

FRAME_HOP_S = 0.010

# Peterson & Barney (1952) male averages: (F1, F2, F3) in Hz.
VOWELS: dict[str, tuple[float, float, float]] = {
    "i": (270, 2290, 3010),
    "I": (390, 1990, 2550),
    "E": (530, 1840, 2480),
    "ae": (660, 1720, 2410),
    "a": (730, 1090, 2440),
    "o": (570, 840, 2410),
    "U": (440, 1020, 2240),
    "u": (300, 870, 2240),
    "V": (640, 1190, 2390),
    "er": (490, 1350, 1690),
}
FORMANT_BW = (70.0, 100.0, 140.0, 250.0, 300.0)
FIXED_F4_F5 = (3500.0, 4500.0)

FRICATIVES = ("s", "sh", "f")
PLOSIVES = ("t", "k", "p")


@dataclass
class VoiceParams:
    sex: str
    f0_mean: float
    f0_range_st: float
    tract_scale: float
    rate: float
    glottal_fc: float
    breathiness: float
    jitter: float
    shimmer: float
    lombard: float
    pause_fraction: float


def random_voice(rng: np.random.Generator, lombard: float = 0.0) -> VoiceParams:
    """Draw a speaker. ``lombard`` in [0, 1] applies the Lombard effect: higher
    F0 (up to ~+50 %), flatter spectral tilt, slightly raised F1, a bit slower."""
    male = rng.random() < 0.5
    f0 = rng.uniform(88.0, 150.0) if male else rng.uniform(170.0, 245.0)
    scale = rng.uniform(0.96, 1.04) * (1.0 if male else rng.uniform(1.14, 1.20))
    return VoiceParams(
        sex="male" if male else "female",
        f0_mean=f0 * (1.0 + 0.5 * lombard * rng.uniform(0.3, 1.0)),
        f0_range_st=rng.uniform(2.0, 5.0) * (1.0 + 0.3 * lombard),
        tract_scale=scale * (1.0 + 0.04 * lombard),
        rate=rng.uniform(0.85, 1.25) * (1.0 - 0.1 * lombard),
        glottal_fc=rng.uniform(250.0, 450.0) * (1.0 + 1.2 * lombard),
        breathiness=rng.uniform(0.005, 0.03),
        jitter=rng.uniform(0.002, 0.008),
        shimmer=rng.uniform(0.02, 0.05),
        lombard=lombard,
        pause_fraction=rng.uniform(0.2, 0.4),
    )


@dataclass
class Segment:
    kind: str  # 'V' vowel, 'F' fricative, 'B' plosive, 'P' pause
    label: str
    t0: float
    t1: float
    # plosive sub-structure (closure silence, burst, aspiration)
    closure: float = 0.0
    burst: float = 0.0
    phrase: int = 0


@dataclass
class SpeechClip:
    wave: np.ndarray  # float32, unit active-speech RMS
    sr: int
    labels: dict = field(default_factory=dict)


# --------------------------------------------------------------- planning


def _plan(duration: float, p: VoiceParams, rng: np.random.Generator) -> list[Segment]:
    segs: list[Segment] = []
    t = rng.uniform(0.05, 0.3)
    phrase = 0
    words_in_phrase = 0
    words_target = int(rng.integers(3, 7))
    rate = p.rate
    speech_time = 0.0
    pause_time = t

    while t < duration:
        for _ in range(int(rng.integers(1, 4))):  # syllables per word
            if rng.random() < 0.75:  # onset consonant
                if rng.random() < 0.5:
                    lab = FRICATIVES[int(rng.integers(0, len(FRICATIVES)))]
                    d = rng.uniform(0.07, 0.14) / rate
                    segs.append(Segment("F", lab, t, t + d, phrase=phrase))
                else:
                    lab = PLOSIVES[int(rng.integers(0, len(PLOSIVES)))]
                    clo, bur, asp = (
                        rng.uniform(0.03, 0.06) / rate,
                        rng.uniform(0.012, 0.025),
                        rng.uniform(0.02, 0.04),
                    )
                    d = clo + bur + asp
                    segs.append(Segment("B", lab, t, t + d, clo, bur, phrase))
                speech_time += d
                t += d
            vlab = list(VOWELS)[int(rng.integers(0, len(VOWELS)))]
            d = rng.uniform(0.08, 0.20) / rate
            segs.append(Segment("V", vlab, t, t + d, phrase=phrase))
            speech_time += d
            t += d
            if rng.random() < 0.25:  # coda
                lab = ("s", "t", "k")[int(rng.integers(0, 3))]
                if lab == "s":
                    d = rng.uniform(0.07, 0.12) / rate
                    segs.append(Segment("F", "s", t, t + d, phrase=phrase))
                else:
                    clo, bur, asp = 0.025 / rate, rng.uniform(0.012, 0.02), 0.015
                    d = clo + bur + asp
                    segs.append(Segment("B", lab, t, t + d, clo, bur, phrase))
                speech_time += d
                t += d
            if t >= duration:
                break
        words_in_phrase += 1
        gap = rng.uniform(0.03, 0.09)
        t += gap
        pause_time += gap
        if words_in_phrase >= words_target:
            pp = rng.uniform(0.25, 0.6)
            t += pp
            pause_time += pp
            phrase += 1
            words_in_phrase = 0
            words_target = int(rng.integers(3, 7))

    # Rescale the inter-phrase pauses so the total pause fraction matches.
    total = max(t, 1e-6)
    cur_frac = pause_time / total
    if abs(cur_frac - p.pause_fraction) > 0.02 and len(segs) > 4:
        last_phrase = max(s.phrase for s in segs)
        scale = np.clip(p.pause_fraction / max(cur_frac, 1e-3), 0.5, 3.0)
        shift = 0.0
        prev_end = segs[0].t0
        out: list[Segment] = []
        cur_phrase = segs[0].phrase
        for s in segs:
            if s.phrase != cur_phrase:
                gap = s.t0 - prev_end
                shift += gap * (scale - 1.0) if gap > 0.2 else 0.0
                cur_phrase = s.phrase
            dur_s = s.t1 - s.t0
            s2 = Segment(
                s.kind,
                s.label,
                s.t0 + shift,
                s.t0 + shift + dur_s,
                s.closure,
                s.burst,
                s.phrase,
            )
            out.append(s2)
            prev_end = s.t1
        segs = out
        del last_phrase
    return [s for s in segs if s.t0 < duration]


# -------------------------------------------------------------- synthesis


def _raised_cosine_env(n: int, ramp: int) -> np.ndarray:
    env = np.ones(n)
    r = min(ramp, n // 2)
    if r > 0:
        ramp_up = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
        env[:r] = ramp_up
        env[n - r :] = ramp_up[::-1]
    return env


def _resonator(f: float, bw: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Klatt 2-pole resonator, unity gain at DC."""
    c = -np.exp(-2.0 * np.pi * bw / sr)
    b = 2.0 * np.exp(-np.pi * bw / sr) * np.cos(2.0 * np.pi * f / sr)
    a = 1.0 - b - c
    return np.array([a]), np.array([1.0, -b, -c])


def _smooth_curve(
    n: int, ctrl_t: np.ndarray, ctrl_v: np.ndarray, sr: int
) -> np.ndarray:
    return np.interp(np.arange(n) / sr, ctrl_t, ctrl_v)


def active_speech_rms(x: np.ndarray, sr: int, thresh_db: float = 30.0) -> float:
    """Long-term RMS over *active* 20 ms frames (frames within ``thresh_db`` of
    the loudest 100 ms-smoothed frame). Used to calibrate speech SPL for both
    synthetic and real speech so the two tiers are directly comparable."""
    x = np.asarray(x, dtype=np.float64)
    frame = int(0.02 * sr)
    n = (x.size // frame) * frame
    if n == 0:
        return float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    fr = x[:n].reshape(-1, frame)
    e = np.mean(fr**2, axis=1) + 1e-20
    e_db = 10 * np.log10(e)
    smooth = np.convolve(e_db, np.ones(5) / 5, mode="same")
    active = smooth > (smooth.max() - thresh_db)
    if not active.any():
        return float(np.sqrt(np.mean(e)))
    return float(np.sqrt(np.mean(e[active])))


def synthesize_speech(
    duration_s: float,
    rng: np.random.Generator,
    sr: int = 48_000,
    lombard: float = 0.0,
    params: VoiceParams | None = None,
) -> SpeechClip:
    p = params if params is not None else random_voice(rng, lombard)
    n = round(duration_s * sr)
    segs = _plan(duration_s, p, rng)
    t_axis = np.arange(n) / sr

    # ---- control curves ------------------------------------------------
    vowels = [s for s in segs if s.kind == "V"]
    voiced_env = np.zeros(n)
    f1 = np.full(n, 500.0)
    f2 = np.full(n, 1500.0)
    f3 = np.full(n, 2500.0)
    ctrl_t = [0.0]
    ctrl_f = [VOWELS["V"]]
    accent_t, accent_v = [0.0], [0.0]
    for s in vowels:
        i0, i1 = int(s.t0 * sr), min(int(s.t1 * sr), n)
        if i1 - i0 < 4:
            continue
        voiced_env[i0:i1] = np.maximum(
            voiced_env[i0:i1], _raised_cosine_env(i1 - i0, int(0.012 * sr))
        )
        tgt = tuple(
            v * p.tract_scale * rng.uniform(0.97, 1.03) for v in VOWELS[s.label]
        )
        ctrl_t += [s.t0 + 0.2 * (s.t1 - s.t0), s.t0 + 0.8 * (s.t1 - s.t0)]
        ctrl_f += [tgt, tgt]
        accent_t.append(0.5 * (s.t0 + s.t1))
        accent_v.append(rng.uniform(-p.f0_range_st / 2, p.f0_range_st / 2))
    ctrl_t.append(duration_s)
    ctrl_f.append(ctrl_f[-1])
    ctrl_arr = np.array(ctrl_f)
    ct = np.array(ctrl_t)
    f1[:] = _smooth_curve(n, ct, ctrl_arr[:, 0], sr)
    f2[:] = _smooth_curve(n, ct, ctrl_arr[:, 1], sr)
    f3[:] = _smooth_curve(n, ct, ctrl_arr[:, 2], sr)

    # F0 contour: phrase declination + accents + slow wobble (semitones).
    accent_t.append(duration_s)
    accent_v.append(0.0)
    accents = _smooth_curve(n, np.array(accent_t), np.array(accent_v), sr)
    phrase_idx = np.zeros(n)
    for s in segs:
        phrase_idx[int(s.t0 * sr) : min(int(s.t1 * sr), n)] = s.phrase
    # fill gaps with the last phrase value
    pos = np.maximum.accumulate(phrase_idx)
    decl = np.zeros(n)
    for ph in np.unique(pos):
        m = pos == ph
        idx = np.nonzero(m)[0]
        if idx.size > 1:
            decl[idx] = np.linspace(1.0, -2.0, idx.size)
    ctrl_w = rng.standard_normal(int(duration_s * 4) + 4)
    wobble = CubicSpline(np.linspace(0, n - 1, ctrl_w.size), ctrl_w)(np.arange(n)) * 0.3
    f0 = p.f0_mean * 2.0 ** ((decl + accents + wobble) / 12.0)
    f0 = np.clip(f0, 60.0, 450.0)
    jit_ctrl = rng.standard_normal(int(duration_s * 200) + 4)
    jit = CubicSpline(np.linspace(0, n - 1, jit_ctrl.size), jit_ctrl)(np.arange(n))
    f0 = f0 * (1.0 + p.jitter * jit)

    # ---- glottal excitation --------------------------------------------
    phase = np.cumsum(f0 / sr)
    wraps = np.floor(phase)
    pulse_idx = np.nonzero(np.diff(wraps, prepend=wraps[0]) > 0)[0]
    exc = np.zeros(n)
    exc[pulse_idx] = 1.0 + p.shimmer * rng.standard_normal(pulse_idx.size)
    breath = p.breathiness * rng.standard_normal(n)
    breath = sps.lfilter(*sps.butter(1, 500.0, "high", fs=sr), breath)
    exc = (exc + breath) * voiced_env

    # glottal flow shaping: two 1-pole low-passes; radiation differentiator.
    a = np.exp(-2.0 * np.pi * p.glottal_fc / sr)
    for _ in range(2):
        exc = sps.lfilter([1.0 - a], [1.0, -a], exc)

    # Time-varying cascade of formant resonators (F1-F3 move, F4/F5 fixed).
    chunk = 96
    voiced = np.zeros(n)
    zi = [np.zeros(2) for _ in range(5)]
    fixed = [
        _resonator(FIXED_F4_F5[0] * p.tract_scale**0.5, FORMANT_BW[3], sr),
        _resonator(FIXED_F4_F5[1] * p.tract_scale**0.5, FORMANT_BW[4], sr),
    ]
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        seg = exc[c0:c1]
        mid = (c0 + c1) // 2
        for j, fj in enumerate((f1[mid], f2[mid], f3[mid])):
            b_, a_ = _resonator(min(fj, 0.45 * sr), FORMANT_BW[j], sr)
            seg, zi[j] = sps.lfilter(b_, a_, seg, zi=zi[j])
        for j, (b_, a_) in enumerate(fixed):
            seg, zi[3 + j] = sps.lfilter(b_, a_, seg, zi=zi[3 + j])
        voiced[c0:c1] = seg
    voiced = sps.lfilter([1.0, -0.97], [1.0], voiced)  # lip radiation

    vmask = voiced_env > 0.05
    if vmask.any():
        voiced /= np.sqrt(np.mean(voiced[vmask] ** 2)) + 1e-12

    # ---- fricatives and plosive bursts ---------------------------------
    wave = voiced.copy()
    lomb = 1.0 + 0.5 * p.lombard
    fric_filters = {
        "s": sps.butter(4, [4500.0, 11000.0], "bandpass", fs=sr, output="sos"),
        "sh": sps.butter(4, [2500.0, 7000.0], "bandpass", fs=sr, output="sos"),
        "f": sps.butter(2, [1500.0, 10000.0], "bandpass", fs=sr, output="sos"),
    }
    fric_gain = {"s": 0.30, "sh": 0.35, "f": 0.08}
    burst_filters = {
        "t": sps.butter(4, [3000.0, 7000.0], "bandpass", fs=sr, output="sos"),
        "k": sps.butter(4, [1500.0, 3500.0], "bandpass", fs=sr, output="sos"),
        "p": sps.butter(2, [400.0, 1800.0], "bandpass", fs=sr, output="sos"),
    }
    active = voiced_env > 0.05
    unvoiced_active = np.zeros(n, dtype=bool)
    for s in segs:
        i0, i1 = int(s.t0 * sr), min(int(s.t1 * sr), n)
        if i1 - i0 < 8:
            continue
        if s.kind == "F":
            noise = sps.sosfilt(fric_filters[s.label], rng.standard_normal(i1 - i0))
            noise /= np.sqrt(np.mean(noise**2)) + 1e-12
            seg = (
                noise
                * fric_gain[s.label]
                * lomb
                * _raised_cosine_env(i1 - i0, int(0.01 * sr))
            )
            wave[i0:i1] += seg
            unvoiced_active[i0:i1] = True
        elif s.kind == "B":
            b0 = i0 + int(s.closure * sr)
            b1 = min(b0 + int(s.burst * sr), n)
            a1 = min(i1, n)
            if b1 - b0 > 4:
                nz = sps.sosfilt(burst_filters[s.label], rng.standard_normal(b1 - b0))
                nz /= np.sqrt(np.mean(nz**2)) + 1e-12
                wave[b0:b1] += (
                    0.5 * lomb * nz * np.exp(-np.arange(b1 - b0) / (0.004 * sr))
                )
            if a1 - b1 > 4:
                nz = sps.sosfilt(fric_filters["sh"], rng.standard_normal(a1 - b1))
                nz /= np.sqrt(np.mean(nz**2)) + 1e-12
                wave[b1:a1] += (
                    0.12 * lomb * nz * _raised_cosine_env(a1 - b1, int(0.005 * sr))
                )
            unvoiced_active[b0:a1] = True

    active = active | unvoiced_active

    # ---- labels ---------------------------------------------------------
    hop = round(FRAME_HOP_S * sr)
    nfr = n // hop
    fr = lambda a: a[: nfr * hop].reshape(nfr, hop)
    f0_frames = np.zeros(nfr)
    vm = fr(voiced_env > 0.5)
    voiced_frames = vm.mean(axis=1) > 0.5
    f0_fr = fr(f0)
    f0_frames[voiced_frames] = np.where(vm, f0_fr, 0.0).sum(axis=1)[
        voiced_frames
    ] / np.maximum(vm.sum(axis=1)[voiced_frames], 1)
    active_frames = fr(active).mean(axis=1) > 0.5
    formants = np.zeros((nfr, 3))
    for j, fj in enumerate((f1, f2, f3)):
        v = fr(fj).mean(axis=1)
        formants[:, j] = np.where(voiced_frames, v, 0.0)
    phones = [
        {
            "start": float(s.t0),
            "end": float(min(s.t1, duration_s)),
            "label": s.label,
            "kind": s.kind,
        }
        for s in segs
    ]

    # Normalise to unit active-speech RMS.
    rms = active_speech_rms(wave, sr)
    wave = (wave / (rms + 1e-12)).astype(np.float32)

    labels = {
        "frame_hop_s": FRAME_HOP_S,
        "f0_hz": f0_frames.astype(np.float32),
        "voiced": voiced_frames,
        "active": active_frames,
        "formants_hz": formants.astype(np.float32),
        "phones": phones,
        "voice": {
            "sex": p.sex,
            "f0_mean": p.f0_mean,
            "tract_scale": p.tract_scale,
            "glottal_fc": p.glottal_fc,
            "lombard": p.lombard,
            "pause_fraction_target": p.pause_fraction,
        },
    }
    del t_axis
    return SpeechClip(wave=wave, sr=sr, labels=labels)
