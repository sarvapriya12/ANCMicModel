"""Synthetic battlefield / aviation noise sources.

Each generator is parametrised from literature-typical ranges and randomised
within them (domain randomisation); none depends on one magic number. Output
is a mono float64 signal at ``sr`` normalised to **unit RMS** for stationary
classes and **unit peak** for impulsive classes. The scene composer assigns
absolute levels (dB SPL).

Sources and their dominant physics (see docs/SYNTHETIC_DATA.md):

jet         jet-mixing noise (broad hump 100 Hz-1 kHz), fan blade-pass tones
            (1-5 kHz + harmonics), "buzz-saw" shaft-order tones, turbine
            whine (3-10 kHz), afterburner crackle
helicopter  main-rotor BPF harmonics (10-30 Hz), blade slap, tail rotor,
            turbine whine
tank        track-pitch harmonics (10-100 Hz), diesel firing-frequency
            harmonics (50-260 Hz, drifting RPM), clank transients
missile     rocket-plume rumble (hump 30-500 Hz), crackle, launch onset
gunfire     Friedlander muzzle blast + supersonic N-wave crack + reverb tail
artillery   long Friedlander blast, low-passed by distance, rumble tail
wind        per-mic independent LF turbulence with gust modulation
babble      competing far-field speech
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps
from scipy.interpolate import CubicSpline

STATIONARY = "stationary"
IMPULSIVE = "impulsive"


@dataclass
class NoiseSource:
    signal: np.ndarray  # unit RMS (stationary) or unit peak (impulsive)
    kind: str
    cls: str


# ---------------------------------------------------------------- helpers


def _unit_rms(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(x**2)))
    return x / r if r > 0 else x


def _scale_rms(x: np.ndarray, rms: float) -> np.ndarray:
    return _unit_rms(x) * rms


def _hump(f: np.ndarray, fc: float, width_oct: float) -> np.ndarray:
    """Gaussian in log-frequency (a smooth spectral hump)."""
    out = np.zeros_like(f)
    pos = f > 0
    out[pos] = np.exp(-0.5 * (np.log2(f[pos] / fc) / width_oct) ** 2)
    return out


def _shaped_noise(
    n: int, sr: int, psd: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Gaussian noise whose power spectral density follows ``psd(f)``."""
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.sqrt(np.maximum(psd(f), 0.0))
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) * mag
    spec[0] = 0.0
    return _unit_rms(np.fft.irfft(spec, n))


def _smooth_random(
    n: int, sr: int, rate_hz: float, rng: np.random.Generator
) -> np.ndarray:
    """Zero-mean, unit-std smooth random curve with ~``rate_hz`` bandwidth."""
    # Random control points every ~1/(2*rate) seconds, joined by a cubic spline:
    # smooth, band-limited to roughly ``rate_hz``, and O(n) to evaluate.
    k = max(int(n / sr * rate_hz * 2) + 4, 4)
    ctrl = rng.standard_normal(k)
    xs = np.linspace(0, n - 1, k)
    y = CubicSpline(xs, ctrl)(np.arange(n))
    s = y.std()
    return (y - y.mean()) / s if s > 0 else y


def _gain_mod(
    n: int, sr: int, rate_hz: float, depth_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Slow multiplicative power fluctuation (amplitude gain curve)."""
    return 10.0 ** (depth_db * _smooth_random(n, sr, rate_hz, rng) / 20.0)


def _phase(f_track: np.ndarray, sr: int) -> np.ndarray:
    return 2.0 * np.pi * np.cumsum(f_track) / sr


def _tone_track(
    n: int, sr: int, f0: float, drift_frac: float, rate_hz: float, rng
) -> np.ndarray:
    return f0 * (1.0 + drift_frac * _smooth_random(n, sr, rate_hz, rng))


def _harmonic_stack(
    f_track: np.ndarray,
    sr: int,
    n_harm: int,
    decay: float,
    rng: np.random.Generator,
    max_hz: float | None = None,
    amp_fn: Callable[[int], float] | None = None,
) -> np.ndarray:
    ph = _phase(f_track, sr)
    out = np.zeros(f_track.size)
    limit = 0.45 * sr if max_hz is None else min(max_hz, 0.45 * sr)
    fmax = float(f_track.max())
    for k in range(1, n_harm + 1):
        if k * fmax > limit:
            break
        a = amp_fn(k) if amp_fn else k ** (-decay)
        out += a * np.sin(k * ph + rng.uniform(0, 2 * np.pi))
    return out


def _place(
    n: int, sr: int, times_s: np.ndarray, amps: np.ndarray, kernel
) -> np.ndarray:
    """Convolve a sparse impulse train with ``kernel`` (truncated to n)."""
    imp = np.zeros(n)
    idx = np.round(times_s * sr).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(imp, idx[ok], np.asarray(amps)[ok])
    return sps.fftconvolve(imp, kernel)[:n]


def friedlander(
    sr: int, t_plus_s: float, b: float = 1.0, length_s: float | None = None
):
    """Friedlander blast wave p(t) = (1 - t/t+) exp(-b t/t+), t >= 0."""
    length = length_s if length_s is not None else max(8 * t_plus_s, 0.004)
    t = np.arange(int(length * sr)) / sr
    return (1.0 - t / t_plus_s) * np.exp(-b * t / t_plus_s)


def n_wave(sr: int, duration_s: float):
    """Ballistic shock (supersonic bullet) N-wave."""
    t = np.arange(max(int(duration_s * sr), 2)) / sr
    return 1.0 - 2.0 * t / duration_s


def damped_resonance(sr: int, f_hz: float, tau_s: float):
    t = np.arange(int(8 * tau_s * sr)) / sr
    return np.sin(2 * np.pi * f_hz * t) * np.exp(-t / tau_s)


def _crackle(
    n: int, sr: int, rate_hz: float, rng: np.random.Generator, sigma: float = 1.0
) -> np.ndarray:
    """Positively skewed pressure crackle: sparse sharp compressions."""
    count = rng.poisson(rate_hz * n / sr)
    times = rng.uniform(0, n / sr, size=count)
    amps = rng.lognormal(0.0, sigma, size=count)
    t = np.arange(int(0.004 * sr)) / sr
    kernel = (1.0 - np.exp(-t / 5e-5)) * np.exp(-t / 3e-4)  # fast rise, short decay
    return _place(n, sr, times, amps, kernel)


def _lowpass(x: np.ndarray, sr: int, fc: float, order: int = 2) -> np.ndarray:
    sos = sps.butter(order, min(fc, 0.45 * sr), btype="low", fs=sr, output="sos")
    return sps.sosfilt(sos, x)


def _peak_norm(x: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(x)))
    return x / p if p > 0 else x


# -------------------------------------------------------------- generators


def jet(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    f1 = rng.uniform(100.0, 400.0)
    f2 = f1 * rng.uniform(4.0, 8.0)

    def psd(f):
        # Humps are specified per octave (constant-percentage bandwidth), so the
        # per-Hz PSD is hump / f.
        s = _hump(f, f1, 1.3) + 0.5 * _hump(f, f2, 1.1) + 1e-2
        return s / np.maximum(f, 1.0) / (1.0 + (f / 9000.0) ** 4)

    mix = _shaped_noise(n, sr, psd, rng) * _gain_mod(
        n, sr, rng.uniform(0.2, 1.0), 1.5, rng
    )
    mix = _unit_rms(mix)

    # Fan blade-pass tone + harmonics.
    bpf = rng.integers(20, 38) * rng.uniform(3000.0, 8000.0) / 60.0
    f_bpf = _tone_track(n, sr, bpf, 0.01, 0.3, rng)
    fan = _scale_rms(
        _harmonic_stack(f_bpf, sr, 3, 1.0, rng) * _gain_mod(n, sr, 1.0, 1.5, rng),
        0.35,
    )

    # Buzz-saw: shaft-order tones, hump-weighted around ~1 kHz.
    shaft = rng.uniform(50.0, 170.0)
    centre = rng.uniform(500.0, 1800.0)
    f_shaft = _tone_track(n, sr, shaft, 0.005, 0.2, rng)
    buzz = _scale_rms(
        _harmonic_stack(
            f_shaft,
            sr,
            int(5000.0 / shaft),
            1.0,
            rng,
            amp_fn=lambda k: float(
                np.exp(-0.5 * (np.log2(k * shaft / centre) / 1.2) ** 2)
            ),
        ),
        0.30,
    )

    # Turbine whine, slight FM.
    fw = rng.uniform(3000.0, 10000.0)
    whine = _scale_rms(np.sin(_phase(_tone_track(n, sr, fw, 0.01, 0.5, rng), sr)), 0.10)

    total = mix + fan + buzz + whine
    if rng.random() < 0.6:  # afterburner / high power crackle
        total = total + _scale_rms(_crackle(n, sr, rng.uniform(300, 800), rng), 0.4)
    return NoiseSource(_unit_rms(total), STATIONARY, "jet")


def helicopter(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    f_main = rng.uniform(4.5, 6.0)
    blades = int(rng.integers(2, 6))
    bpf = float(np.clip(f_main * blades, 10.0, 30.0))
    f_track = _tone_track(n, sr, bpf, 0.01, 0.2, rng)
    rotor = _scale_rms(_harmonic_stack(f_track, sr, 80, 1.2, rng), 1.0)

    # Blade slap: impulsive at ~BPF, amplitude-modulated.
    period = 1.0 / bpf
    times = np.arange(rng.uniform(0, period), n / sr, period)
    times = times + rng.normal(0, 0.03 * period, times.size)
    am = np.clip(1.0 + 0.5 * _smooth_random(times.size, 10, 1.0, rng), 0.1, None)
    slap = _scale_rms(
        _place(
            n,
            sr,
            times,
            am,
            damped_resonance(sr, rng.uniform(300.0, 1200.0), 3e-3),
        ),
        0.8,
    )

    f_tail = _tone_track(n, sr, rng.uniform(50.0, 120.0), 0.02, 0.3, rng)
    tail = _scale_rms(_harmonic_stack(f_tail, sr, 8, 0.7, rng), 0.4)
    fw = rng.uniform(3000.0, 8000.0)
    whine = _scale_rms(np.sin(_phase(_tone_track(n, sr, fw, 0.01, 0.5, rng), sr)), 0.15)
    bb_fc = rng.uniform(100.0, 400.0)
    bb = _scale_rms(
        _shaped_noise(
            n, sr, lambda f: (_hump(f, bb_fc, 1.5) + 1e-2) / np.maximum(f, 1.0), rng
        ),
        0.7,
    )

    return NoiseSource(
        _unit_rms(rotor + slap + tail + whine + bb), STATIONARY, "helicopter"
    )


def tank(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    # Track pitch: v/p = 10-100 Hz with many harmonics.
    f_track_hz = rng.uniform(10.0, 100.0)
    tr = _tone_track(n, sr, f_track_hz, 0.05, 0.3, rng)
    tracks = _scale_rms(_harmonic_stack(tr, sr, 60, 0.9, rng, max_hz=1500.0), 0.8)

    # Diesel firing frequency n_cyl * RPM / 120 = 50-260 Hz, drifting RPM.
    f_fire = rng.uniform(50.0, 260.0)
    fire_track = _tone_track(n, sr, f_fire, 0.05, 0.4, rng)
    diesel = _scale_rms(
        _harmonic_stack(fire_track, sr, 20, 0.8, rng, max_hz=3000.0), 1.0
    )
    # Second, cross-bank stack slightly detuned.
    diesel2 = _scale_rms(
        _harmonic_stack(fire_track * rng.uniform(1.01, 1.04), sr, 12, 0.9, rng), 0.4
    )

    # Clank transients: random damped resonances (hull / road wheels).
    rate = rng.uniform(2.0, 6.0)
    count = rng.poisson(rate * n / sr)
    times = rng.uniform(0, n / sr, count)
    amps = rng.lognormal(0.0, 0.5, count)
    clank = _place(
        n,
        sr,
        times,
        amps,
        damped_resonance(sr, rng.uniform(300.0, 2000.0), rng.uniform(5e-3, 2e-2)),
    )
    clank = _scale_rms(clank, 0.5) if clank.any() else clank

    # Low-frequency broadband (intake / engine bay), roughly 1/f below 500 Hz.
    lf = _scale_rms(
        _shaped_noise(
            n,
            sr,
            lambda f: (
                1.0 / (1.0 + (np.maximum(f, 1.0) / 60.0)) / (1 + (f / 500.0) ** 4)
            ),
            rng,
        ),
        0.8,
    )
    return NoiseSource(
        _unit_rms(tracks + diesel + diesel2 + clank + lf), STATIONARY, "tank"
    )


def missile(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    fp = rng.uniform(30.0, 500.0)

    def psd(f):
        per_octave = _hump(f, fp, 1.4) / (1.0 + (f / (4.0 * fp)) ** 2) + 1e-2 / (
            1.0 + (f / 6000.0) ** 2
        )
        return per_octave / np.maximum(f, 1.0)

    rumble = _shaped_noise(n, sr, psd, rng) * _gain_mod(n, sr, 0.5, 2.0, rng)
    total = _unit_rms(rumble) + _scale_rms(
        _crackle(n, sr, rng.uniform(300.0, 1500.0), rng), 0.5
    )
    # Launch onset (steady-state half of the time, ramp-up otherwise).
    if rng.random() < 0.5:
        t0 = rng.uniform(0.0, 0.4) * n / sr
        ramp = rng.uniform(0.3, 2.0)
        t = np.arange(n) / sr
        env = np.clip((t - t0) / ramp, 0.0, 1.0)
        total = total * (0.03 + 0.97 * env**1.5)
    return NoiseSource(_unit_rms(total), STATIONARY, "missile")


def gunfire(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    dur = n / sr
    times: list[float] = []
    if rng.random() < 0.5:  # isolated shots
        times = list(
            np.sort(rng.uniform(0.0, dur, rng.poisson(rng.uniform(0.3, 2.0) * dur) + 1))
        )
    else:  # bursts at 600-900 rpm
        t = rng.uniform(0.0, dur * 0.5)
        while t < dur:
            cyc = 1.0 / rng.uniform(10.0, 15.0)
            for k in range(int(rng.integers(3, 9))):
                times.append(t + k * cyc)
            t += rng.uniform(1.5, 4.0)
    times_a = np.array([t for t in times if t < dur])
    if times_a.size == 0:
        times_a = np.array([dur * 0.3])
    amps = rng.lognormal(0.0, 0.3, times_a.size)

    t_plus = rng.uniform(3e-4, 3e-3)
    blast = _place(
        n, sr, times_a + rng.uniform(0.003, 0.04), amps, friedlander(sr, t_plus)
    )
    blast = _lowpass(blast, sr, rng.uniform(3000.0, 6000.0))
    out = blast
    if rng.random() < 0.7:  # supersonic crack precedes the blast
        crack = _place(n, sr, times_a, 0.8 * amps, n_wave(sr, rng.uniform(1e-4, 3e-4)))
        out = out + crack
    # Reverberant tail: decaying band-limited noise triggered by each shot.
    rt = rng.uniform(0.2, 1.0)
    tt = np.arange(int(min(rt * 1.5, 1.2) * sr)) / sr
    tail_kernel = rng.standard_normal(tt.size) * np.exp(-6.9 * tt / rt)
    tail_kernel = _lowpass(tail_kernel, sr, 2500.0)
    out = out + rng.uniform(0.05, 0.15) * sps.fftconvolve(
        _place(n, sr, times_a + 0.01, amps * 0.2, np.array([1.0])), tail_kernel
    )[:n] / max(np.abs(tail_kernel).max(), 1e-9)
    return NoiseSource(_peak_norm(out), IMPULSIVE, "gunfire")


def artillery(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    dur = n / sr
    times = np.sort(rng.uniform(0.0, dur * 0.8, int(rng.integers(1, 4))))
    amps = rng.lognormal(0.0, 0.3, times.size)
    t_plus = rng.uniform(5e-3, 5e-2)
    blast = _place(n, sr, times, amps, friedlander(sr, t_plus, b=rng.uniform(1.0, 2.0)))
    blast = _lowpass(blast, sr, rng.uniform(100.0, 300.0), order=2)
    rt = rng.uniform(1.0, 3.0)
    tt = np.arange(int(min(rt, 2.5) * sr)) / sr
    kernel = _lowpass(
        rng.standard_normal(tt.size), sr, rng.uniform(80.0, 200.0)
    ) * np.exp(-6.9 * tt / rt)
    rumble = (
        0.3
        * sps.fftconvolve(_place(n, sr, times, amps, np.array([1.0])), kernel)[:n]
        / max(np.abs(kernel).max(), 1e-9)
    )
    return NoiseSource(_peak_norm(blast + rumble), IMPULSIVE, "artillery")


def wind(n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    """One mic's wind/rotor-wash turbulence. Call once *per mic* with different
    generators: wind noise is local to each capsule and incoherent."""
    fc = rng.uniform(100.0, 500.0)

    def psd(f):
        return 1.0 / (1.0 + (np.maximum(f, 1.0) / fc) ** 3)

    base = _shaped_noise(n, sr, psd, rng)
    depth = rng.uniform(0.5, 1.0)
    gust = np.clip(
        1.0 + depth * _smooth_random(n, sr, rng.uniform(0.1, 2.0), rng), 0.05, None
    )
    return NoiseSource(_unit_rms(base * gust), STATIONARY, "wind")


def babble(clips: list[np.ndarray], n: int, rng: np.random.Generator) -> NoiseSource:
    """Competing far-field speech from ``clips`` (random segments, random gains)."""
    out = np.zeros(n)
    for clip in clips:
        clip = np.asarray(clip, dtype=np.float64)
        if clip.size < n:
            clip = np.tile(clip, int(np.ceil(n / max(clip.size, 1))))
        start = int(rng.integers(0, clip.size - n + 1))
        seg = clip[start : start + n]
        out += _unit_rms(seg) * 10.0 ** (rng.uniform(-6.0, 0.0) / 20.0)
    return NoiseSource(_unit_rms(out), STATIONARY, "babble")


GENERATORS: dict[str, Callable[[int, int, np.random.Generator], NoiseSource]] = {
    "jet": jet,
    "helicopter": helicopter,
    "tank": tank,
    "missile": missile,
    "gunfire": gunfire,
    "artillery": artillery,
    "wind": wind,
}

# Classes allowed as the scene's continuous / dominant noise, and impulsive ones.
CONTINUOUS_CLASSES = ("jet", "helicopter", "tank", "missile")
IMPULSIVE_CLASSES = ("gunfire", "artillery")
ALL_CLASSES = tuple(GENERATORS)


def make_noise(cls: str, n: int, sr: int, rng: np.random.Generator) -> NoiseSource:
    try:
        gen = GENERATORS[cls]
    except KeyError as exc:
        raise ValueError(
            f"unknown noise class {cls!r}; available: {sorted(GENERATORS)}"
        ) from exc
    src = gen(n, sr, rng)
    if not np.all(np.isfinite(src.signal)):
        raise FloatingPointError(f"noise generator {cls!r} produced non-finite output")
    return src
