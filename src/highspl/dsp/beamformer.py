"""Geometry-aware stage-1 front-ends for the back-to-back, near-field pod.

Geometry (see data/rir.py): primary mic ``f`` faces the lips (r1 ~ 5.5 cm), the
reference mic ``r`` sits directly behind it at spacing d (~1.5 cm), facing away.
With tau = d/c (~2.1 samples at 48 kHz):

  far-field wave from the front :  r(t) =       f(t - tau)
  far-field wave from the rear  :  r(t) =       f(t + tau)
  mouth (near field)            :  r(t) =  g * f(t - tau),  g = r1/(r1+d) ~ 0.79

Three candidates (all block-vectorised, ~tens of MAC/sample, no look-ahead):

diff   D3  fixed first-order differential beamformer: forward cardioid
           c_F = f - r(t-tau) (null behind), backward cardioid c_B = r - f(t-tau),
           y = c_F - beta * c_B, followed by a bounded minimum-phase EQ that
           flattens the *mouth* response (near-field LF is otherwise -13 dB).
adiff  D4  same, with beta adapted (Elko) to null the strongest rear-half noise.
gsc    D5  generalised sidelobe canceller: D3 as the fixed beam, a near-field
           *blocking* reference u = r - g_eff * f(t-tau) that cancels the mouth
           exactly, and a constrained block-NLMS canceller that subtracts the
           noise that is coherent with u. Adapts only when speech is absent.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .frontend import register_frontend

C_SOUND = 343.0
N_TAPS = 19
DELAY = 8  # integer delay applied to both paths (centre of the FIR)


def fractional_delay_fir(
    delay: float, n_taps: int = N_TAPS, half_width: float = 9.0
) -> np.ndarray:
    """Kaiser-windowed sinc delaying by ``DELAY + delay`` samples."""
    c = DELAY + delay
    n = np.arange(n_taps)
    x = n - c
    h = np.sinc(x)
    arg = np.clip(1.0 - (x / half_width) ** 2, 0.0, None)
    h *= np.i0(6.0 * np.sqrt(arg)) / np.i0(6.0)
    h[np.abs(x) > half_width] = 0.0
    return h / h.sum()


def mouth_response(
    freqs: np.ndarray, tau_samples: float, g: float, beta: float, fs: int
) -> np.ndarray:
    """Complex response of ``y = c_F - beta*c_B`` to the near-field mouth."""
    w = 2.0 * np.pi * freqs / fs
    return (1.0 - g * np.exp(-2j * w * tau_samples)) + beta * (1.0 - g) * np.exp(
        -1j * w * tau_samples
    )


def _shelf_sos(fs: int, f0: float, gain_db: float, low: bool) -> np.ndarray:
    """RBJ shelving biquad as one second-order section (stable for any f0, gain)."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cosw, sinw = np.cos(w0), np.sin(w0)
    alpha = sinw / 2.0 * np.sqrt(2.0)
    sq = 2.0 * np.sqrt(a_lin) * alpha
    if low:
        b0 = a_lin * ((a_lin + 1) - (a_lin - 1) * cosw + sq)
        b1 = 2 * a_lin * ((a_lin - 1) - (a_lin + 1) * cosw)
        b2 = a_lin * ((a_lin + 1) - (a_lin - 1) * cosw - sq)
        a0 = (a_lin + 1) + (a_lin - 1) * cosw + sq
        a1 = -2 * ((a_lin - 1) + (a_lin + 1) * cosw)
        a2 = (a_lin + 1) + (a_lin - 1) * cosw - sq
    else:
        b0 = a_lin * ((a_lin + 1) + (a_lin - 1) * cosw + sq)
        b1 = -2 * a_lin * ((a_lin - 1) + (a_lin + 1) * cosw)
        b2 = a_lin * ((a_lin + 1) + (a_lin - 1) * cosw - sq)
        a0 = (a_lin + 1) - (a_lin - 1) * cosw + sq
        a1 = 2 * ((a_lin - 1) - (a_lin + 1) * cosw)
        a2 = (a_lin + 1) - (a_lin - 1) * cosw - sq
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


def design_mouth_eq(
    tau_samples: float,
    g: float,
    beta: float,
    fs: int,
    max_gain_db: float = 12.0,
) -> np.ndarray:
    """Three shelving biquads (SOS) that flatten the mouth response.

    Least-squares fit of the dB response of (EQ x mouth response) to flat over
    80 Hz-8 kHz, with every shelf gain bounded by ``max_gain_db``. The mouth
    response is only -13 dB at low frequencies (the pod is a pressure-gradient
    pair), so an unbounded EQ would also boost low-frequency self-noise."""
    from scipy.optimize import least_squares

    f = np.geomspace(80.0, 8000.0, 80)
    m_db = 20.0 * np.log10(np.abs(mouth_response(f, tau_samples, g, beta, fs)) + 1e-9)
    ref_db = 20.0 * np.log10(
        np.abs(mouth_response(np.array([2000.0]), tau_samples, g, beta, fs))[0]
    )

    def build(p):
        return np.vstack(
            [
                _shelf_sos(fs, p[0], p[1], low=True),
                _shelf_sos(fs, p[2], p[3], low=True),
                _shelf_sos(fs, p[4], p[5], low=False),
            ]
        )

    def resid(p):
        _, h = sps.sosfreqz(build(p), worN=f, fs=fs)
        return 20.0 * np.log10(np.abs(h) + 1e-9) + m_db - ref_db

    x0 = [500.0, max_gain_db * 0.6, 150.0, max_gain_db * 0.3, 3000.0, -2.0]
    lo = [100.0, 0.0, 50.0, 0.0, 1000.0, -12.0]
    hi = [3000.0, max_gain_db, 1000.0, max_gain_db, 12000.0, 6.0]
    sol = least_squares(resid, x0, bounds=(lo, hi))
    return build(sol.x)


class _Fir:
    """Streaming FIR with state carried across blocks."""

    def __init__(self, b: np.ndarray) -> None:
        self.b = np.asarray(b, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.zi = np.zeros(self.b.size - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sps.lfilter(self.b, [1.0], x, zi=self.zi)
        return y


class _Sos:
    """Streaming second-order-section filter with state carried across blocks."""

    def __init__(self, sos: np.ndarray) -> None:
        self.sos = np.asarray(sos, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.zi = np.zeros((self.sos.shape[0], 2))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sps.sosfilt(self.sos, x, zi=self.zi)
        return y


class _IntDelay:
    def __init__(self, n: int) -> None:
        self.n = n
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.n)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ext = np.concatenate((self.hist, x))
        self.hist = ext[-self.n :] if self.n else ext[:0]
        return ext[: x.size]


class DifferentialFrontEnd:
    """D3 (fixed beta) / D4 (adaptive beta) differential beamformer."""

    def __init__(
        self,
        mic_spacing_m: float = 0.015,
        mouth_distance_m: float = 0.055,
        sample_rate: int = 48_000,
        beta: float = 0.25,
        adaptive: bool = False,
        beta_max: float = 0.7,
        eq_max_gain_db: float = 12.0,
        sub_block: int = 128,
    ) -> None:
        self.name = "adiff" if adaptive else "diff"
        self.fs = sample_rate
        self.tau = mic_spacing_m / C_SOUND * sample_rate
        self.g = mouth_distance_m / (mouth_distance_m + mic_spacing_m)
        self.beta0 = beta
        self.adaptive = adaptive
        self.beta_max = beta_max
        self.sub = sub_block

        self._frac = fractional_delay_fir(self.tau)
        self._eq_sos = design_mouth_eq(
            self.tau, self.g, beta, sample_rate, eq_max_gain_db
        )
        self.reset()

    def reset(self) -> None:
        self._f_d = _IntDelay(DELAY)
        self._r_d = _IntDelay(DELAY)
        self._f_t = _Fir(self._frac)
        self._r_t = _Fir(self._frac)
        self._eq = _Sos(self._eq_sos)
        self.beta = self.beta0
        self._corr = 0.0
        self._pow = 1e-12

    def beams(
        self, f: np.ndarray, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(c_F, c_B, f_aligned, r_aligned): all at delay ``DELAY``."""
        f = np.asarray(f, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        f_d, r_d = self._f_d(f), self._r_d(r)
        f_t, r_t = self._f_t(f), self._r_t(r)
        return f_d - r_t, r_d - f_t, f_d, r_d

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        c_f, c_b, _, _ = self.beams(primary, reference)
        if self.adaptive:
            beta = np.empty(c_f.size)
            for s in range(0, c_f.size, self.sub):
                e = slice(s, s + self.sub)
                # leaky running correlation of the two beams (Elko null steering)
                self._corr = 0.9 * self._corr + float(np.dot(c_f[e], c_b[e]))
                self._pow = 0.9 * self._pow + float(np.dot(c_b[e], c_b[e]))
                self.beta = float(
                    np.clip(self._corr / (self._pow + 1e-12), 0.0, self.beta_max)
                )
                beta[e] = self.beta
            y = c_f - beta * c_b
        else:
            y = c_f - self.beta0 * c_b
        return self._eq(y).astype(np.float32)

    @property
    def delay_samples(self) -> int:
        return DELAY

    def adaptation_frozen(self) -> bool:
        return False


class GSCFrontEnd:
    """D5: fixed differential beam + near-field blocking + constrained canceller."""

    name = "gsc"

    def __init__(
        self,
        mic_spacing_m: float = 0.015,
        mouth_distance_m: float = 0.055,
        sample_rate: int = 48_000,
        beta: float = 0.25,
        taps: int = 32,
        mu: float = 0.3,
        sub_block: int = 64,
        w_l1_max: float = 6.0,
        leakage: float = 1e-4,
        speech_margin_db: float = 3.0,
        blocker_gain: float | None = None,
        clip_threshold: float = 0.98,
        clip_freeze_fraction: float = 0.01,
    ) -> None:
        self.dbf = DifferentialFrontEnd(
            mic_spacing_m, mouth_distance_m, sample_rate, beta=beta, adaptive=False
        )
        self.g = blocker_gain if blocker_gain is not None else self.dbf.g
        self.taps = taps
        self.mu = mu
        self.sub = sub_block
        self.w_l1_max = w_l1_max
        self.leakage = leakage
        self.margin = 10.0 ** (speech_margin_db / 10.0)
        self.clip_thr = clip_threshold
        self.clip_frac = clip_freeze_fraction
        self.d_align = taps // 2  # delay the beam so w can use "future" u samples
        self.reset()

    def reset(self) -> None:
        self.dbf.reset()
        self.w = np.zeros(self.taps)
        self._b_del = _IntDelay(self.d_align)
        self._u_ext = np.zeros(self.taps - 1)
        self._ratio_floor = 1e6
        self._frozen = False
        self._frozen_frac = 0.0

    @property
    def delay_samples(self) -> int:
        return DELAY + self.d_align

    def set_blocker_gain(self, g: float) -> None:
        self.g = float(g)

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        d = self.dbf
        # A clipped capture is non-linear: no linear canceller can model it, so
        # adaptation is frozen for the block (the beam and stale weights still run).
        clipped = (
            float(np.mean(np.abs(primary) >= self.clip_thr)) > self.clip_frac
            or float(np.mean(np.abs(reference) >= self.clip_thr)) > self.clip_frac
        )
        c_f, c_b, _f_d, r_d = d.beams(primary, reference)
        # fixed beam with mouth EQ
        b = d._eq(c_f - d.beta0 * c_b)
        # Noise reference with the mouth blocked exactly:
        #   u = r - g * f(t - tau),  and  f(t - tau) = r_d - c_B  (from c_B = r - f(t-tau))
        f_tau = r_d - c_b
        u = r_d - self.g * f_tau

        b_a = self._b_del(b)  # beam delayed by taps/2; u is not
        u_a = u

        out = np.empty_like(b_a)
        frozen_blocks = 0
        total_blocks = 0
        for s in range(0, b_a.size, self.sub):
            e = slice(s, s + self.sub)
            ub = u_a[e]
            bb = b_a[e]
            ext = np.concatenate((self._u_ext, ub))
            y_hat = np.convolve(ext, self.w, mode="valid")
            err = bb - y_hat
            out[e] = err
            self._u_ext = ext[-(self.taps - 1) :]

            # speech-presence: beam/reference power ratio rises above its floor
            pb = float(np.dot(b[e], b[e])) + 1e-12
            pu = float(np.dot(u[e], u[e])) + 1e-12
            ratio = pb / pu
            # minimum-statistics floor: drops to new minima, rises slowly
            self._ratio_floor = min(self._ratio_floor * 1.0005, ratio)
            speech = ratio > self.margin * self._ratio_floor
            total_blocks += 1
            if clipped or speech or pu < 1e-10:
                frozen_blocks += 1
                continue

            # block NLMS update on the post-cancellation error
            grad = np.correlate(ext, err, mode="valid")[::-1]
            p_in = self.taps * pu / ub.size + 1e-12
            self.w = (1.0 - self.leakage) * self.w + self.mu * grad / ub.size / p_in
            l1 = float(np.abs(self.w).sum())
            if l1 > self.w_l1_max:
                self.w *= self.w_l1_max / l1
        self._frozen_frac = frozen_blocks / max(total_blocks, 1)
        self._frozen = self._frozen_frac > 0.5
        return out.astype(np.float32)

    def adaptation_frozen(self) -> bool:
        return self._frozen


class BlendedFrontEnd:
    """Fade a beamformer in with the noise level.

    The bench showed the differential/GSC processing costs a little speech
    fidelity when there is nothing to remove (SI-SDR 9 vs 14 dB at 90-105 dB
    SPL) but is clearly better at 105-130 dB. The noise floor of the primary mic
    (10th percentile of block power over ~4 s: pauses carry only noise) steers
    ``alpha`` from 0 (primary mic only, delay-aligned) to 1 (full beamformer)
    between ``lo_dbfs`` and ``hi_dbfs``. ``alpha`` is slew-limited so the blend
    never clicks."""

    def __init__(
        self,
        inner,
        lo_dbfs: float = -34.0,
        hi_dbfs: float = -24.0,
        block: int = 512,
        sample_rate: int = 48_000,
        slew_per_block: float = 0.1,
    ) -> None:
        from ..telemetry import NoiseFloorTracker

        self.inner = inner
        self.name = inner.name + "+blend"
        self.lo, self.hi = lo_dbfs, hi_dbfs
        self.slew = slew_per_block
        self._tracker_cls = NoiseFloorTracker
        self._block, self._sr = block, sample_rate
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self._align = _IntDelay(self.inner.delay_samples)
        self._tracker = self._tracker_cls(4.0, self._block, self._sr)
        self.alpha = 1.0  # unknown noise level: assume the harder case

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        self._tracker.update(primary)
        floor = self._tracker.floor_dbfs
        target = (
            1.0
            if not np.isfinite(floor)
            else float(np.clip((floor - self.lo) / (self.hi - self.lo), 0.0, 1.0))
        )
        a0 = self.alpha
        self.alpha = a0 + float(np.clip(target - a0, -self.slew, self.slew))
        y = self.inner.process(primary, reference)
        f = self._align(np.asarray(primary, dtype=np.float64)).astype(np.float32)
        ramp = np.linspace(a0, self.alpha, y.size, endpoint=False, dtype=np.float32)
        return (1.0 - ramp) * f + ramp * y

    def adaptation_frozen(self) -> bool:
        return self.inner.adaptation_frozen()


def _make_diff(**kw) -> DifferentialFrontEnd:
    return DifferentialFrontEnd(adaptive=False, **kw)


def _make_adiff(**kw) -> DifferentialFrontEnd:
    return DifferentialFrontEnd(adaptive=True, **kw)


def _make_gsc(**kw) -> GSCFrontEnd:
    return GSCFrontEnd(**kw)


register_frontend("diff", _make_diff)
register_frontend("adiff", _make_adiff)
register_frontend("gsc", _make_gsc)
