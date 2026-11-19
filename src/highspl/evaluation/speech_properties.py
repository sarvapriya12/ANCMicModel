"""Speech-property preservation scorecard.

Question answered: *do the properties of the (synthetic or real) speech survive
to the output?* Every metric compares an enhanced signal ``est`` with the clean
target ``ref`` (both aligned, 48 kHz in) and is **level-invariant**, so AGC or
gain changes cannot game it.

metric            what it checks                                   ideal
----------------  -----------------------------------------------  -------
f0_recall         ref-voiced frames kept voiced with F0 within 20%  1
f0_gpe            gross pitch errors among frames voiced in both    0
f0_rmse_cents     fine pitch error (frames within 20%)              0
vuv_acc           voiced/unvoiced agreement on active frames        1
ltass_rmse_db     long-term spectrum deviation 300 Hz-4 kHz         0
ltass_max_db      worst 1/3-octave deviation 125 Hz-6.3 kHz         0
fric_db           4-8 kHz energy kept in unvoiced active frames     0
mod_corr          2-8 Hz syllabic-envelope correlation (4 bands)    1
pause_db          residual level in pauses re active speech level   -inf

All analysis runs at 16 kHz with a 10 ms hop.
"""

from __future__ import annotations

import numpy as np
import soxr
from scipy import signal as sps

ANALYSIS_SR = 16_000
HOP = 160  # 10 ms
WIN = 640  # 40 ms

THIRD_OCT = np.array(
    [
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
    ],
    dtype=float,
)


def to_analysis_rate(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if sr == ANALYSIS_SR:
        return x
    return soxr.resample(x, sr, ANALYSIS_SR)


def _frames(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    n = 1 + max(x.size - win, 0) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# --------------------------------------------------------------------- F0


def f0_track(
    x16: np.ndarray, fmin: float = 70.0, fmax: float = 400.0, v_thresh: float = 0.45
) -> dict[str, np.ndarray]:
    """Frame-wise F0 by normalised autocorrelation. Returns f0 (0=unvoiced),
    voicing strength and frame energy (dB)."""
    fr = _frames(x16)
    fr = fr - fr.mean(axis=1, keepdims=True)
    win = np.hanning(WIN)
    fr = fr * win
    nfft = 2048
    spec = np.fft.rfft(fr, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, : WIN // 2]
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, nfft)) ** 2, nfft)[: WIN // 2]
    ac = ac / (wac[None, :] + 1e-12)
    r = ac / (ac[:, :1] + 1e-12)

    lo, hi = int(ANALYSIS_SR / fmax), int(ANALYSIS_SR / fmin)
    seg = r[:, lo : hi + 1]
    peak_val = seg.max(axis=1)
    # Take the first strong local maximum (avoids sub-harmonic octave errors).
    f0 = np.zeros(fr.shape[0])
    strength = np.zeros(fr.shape[0])
    for i in range(fr.shape[0]):
        s = seg[i]
        cand = np.nonzero(
            (s[1:-1] >= s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] >= 0.9 * peak_val[i])
        )[0]
        if cand.size == 0:
            continue
        k = cand[0] + 1
        # parabolic refinement
        a, b, c = s[k - 1], s[k], s[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        lag = lo + k + off
        f0[i] = ANALYSIS_SR / lag
        strength[i] = b
    energy_db = 10 * np.log10(np.mean(_frames(x16) ** 2, axis=1) + 1e-20)
    gate = energy_db > energy_db.max() - 40.0
    voiced = (strength > v_thresh) & gate
    f0 = np.where(voiced, f0, 0.0)
    return {"f0": f0, "strength": strength, "voiced": voiced, "energy_db": energy_db}


def activity_mask(ref16: np.ndarray, thresh_db: float = 35.0) -> np.ndarray:
    """Speech-active frames of the clean target (energy within ``thresh_db`` of
    the loudest smoothed frame)."""
    e = 10 * np.log10(np.mean(_frames(ref16) ** 2, axis=1) + 1e-20)
    sm = np.convolve(e, np.ones(5) / 5, mode="same")
    return sm > sm.max() - thresh_db


# -------------------------------------------------------------- spectrum


def _band_power_frames(x16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrogram on the 40 ms / 10 ms grid -> (freqs, power[frames,bins])."""
    fr = _frames(x16) * np.hanning(WIN)
    p = np.abs(np.fft.rfft(fr, 1024, axis=1)) ** 2
    return np.fft.rfftfreq(1024, 1.0 / ANALYSIS_SR), p


def _third_octave_edges() -> tuple[np.ndarray, np.ndarray]:
    return THIRD_OCT / 2 ** (1 / 6), THIRD_OCT * 2 ** (1 / 6)


def _band_levels(f: np.ndarray, p_mean: np.ndarray) -> np.ndarray:
    lo, hi = _third_octave_edges()
    out = np.zeros(THIRD_OCT.size)
    for i, (a, b) in enumerate(zip(lo, hi)):
        out[i] = p_mean[(f >= a) & (f < min(b, 7990.0))].sum()
    return 10 * np.log10(out + 1e-20)


def _norm_offset_db(f: np.ndarray, p_ref: np.ndarray, p_est: np.ndarray) -> float:
    m = (f >= 300) & (f <= 3400)
    return 10 * np.log10((p_est[m].sum() + 1e-20) / (p_ref[m].sum() + 1e-20))


# --------------------------------------------------------------- envelope


def _band_envelopes(x16: np.ndarray) -> list[np.ndarray]:
    out = []
    for lo, hi in ((250, 500), (500, 1000), (1000, 2000), (2000, 4000)):
        sos = sps.butter(4, [lo, hi], "bandpass", fs=ANALYSIS_SR, output="sos")
        y = sps.sosfilt(sos, x16)
        env = np.sqrt(
            np.maximum(np.convolve(y**2, np.ones(HOP * 2) / (HOP * 2), mode="same"), 0)
        )
        out.append(env[::HOP])
    return out


def modulation_correlation(ref16: np.ndarray, est16: np.ndarray) -> float:
    """Mean Pearson correlation of 2-8 Hz band-passed log-envelopes over four
    octave bands (the syllabic rhythm of speech)."""
    sos = sps.butter(2, [2.0, 8.0], "bandpass", fs=100.0, output="sos")
    cs = []
    for er, ee in zip(_band_envelopes(ref16), _band_envelopes(est16)):
        n = min(er.size, ee.size)
        a = sps.sosfiltfilt(sos, 20 * np.log10(er[:n] + 1e-7))
        b = sps.sosfiltfilt(sos, 20 * np.log10(ee[:n] + 1e-7))
        if a.std() < 1e-9 or b.std() < 1e-9:
            cs.append(0.0)
        else:
            cs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cs))


# ----------------------------------------------------------------- scorecard


def compute_properties(
    ref: np.ndarray, est: np.ndarray, sr: int = 48_000
) -> dict[str, float]:
    """All property metrics for aligned ``ref`` (clean) and ``est`` signals."""
    n = min(ref.size, est.size)
    r16 = to_analysis_rate(ref[:n], sr)
    e16 = to_analysis_rate(est[:n], sr)
    m = min(r16.size, e16.size)
    r16, e16 = r16[:m], e16[:m]

    tr = f0_track(r16)
    te = f0_track(e16)
    act = activity_mask(r16)
    k = min(tr["f0"].size, te["f0"].size, act.size)
    f0r, f0e = tr["f0"][:k], te["f0"][:k]
    vr, ve, act = tr["voiced"][:k], te["voiced"][:k], act[:k]

    # --- F0 / voicing ---
    ref_v = vr & act
    both = ref_v & ve
    rel_err = np.abs(f0e[both] / f0r[both] - 1.0) if both.any() else np.array([])
    ok = rel_err <= 0.2
    f0_recall = float(ok.sum() / ref_v.sum()) if ref_v.any() else np.nan
    f0_gpe = float(np.mean(~ok)) if both.any() else np.nan
    cents = 1200 * np.log2(f0e[both][ok] / f0r[both][ok]) if ok.any() else np.array([])
    f0_rmse = float(np.sqrt(np.mean(cents**2))) if cents.size else np.nan
    vuv = float(np.mean(vr[act] == ve[act])) if act.any() else np.nan

    # --- long-term spectrum / fricatives ---
    f, pr = _band_power_frames(r16)
    _, pe = _band_power_frames(e16)
    kk = min(pr.shape[0], pe.shape[0], act.size)
    pr, pe, act_s = pr[:kk], pe[:kk], act[:kk]
    mean_r = pr[act_s].mean(axis=0) if act_s.any() else pr.mean(axis=0)
    mean_e = pe[act_s].mean(axis=0) if act_s.any() else pe.mean(axis=0)
    off = _norm_offset_db(f, mean_r, mean_e)
    dev = _band_levels(f, mean_e) - off - _band_levels(f, mean_r)
    mid = (THIRD_OCT >= 300) & (THIRD_OCT <= 4000)
    ltass_rmse = float(np.sqrt(np.mean(dev[mid] ** 2)))
    ltass_max = float(np.max(np.abs(dev)))

    hf = (f >= 4000) & (f < 7990)
    unv = act_s & ~vr[:kk]
    if unv.sum() >= 5:
        e_r = pr[unv][:, hf].sum()
        e_e = pe[unv][:, hf].sum()
        fric = float(10 * np.log10((e_e + 1e-20) / (e_r + 1e-20)) - off)
    else:
        fric = np.nan

    # --- pause residual ---
    pad = 10  # keep 100 ms away from speech edges
    dil = np.convolve(act_s.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    pause = ~dil
    en = np.mean(_frames(e16)[:kk] ** 2, axis=1)
    if pause.sum() >= 5 and act_s.any():
        pause_db = float(
            10 * np.log10((en[pause].mean() + 1e-20) / (en[act_s].mean() + 1e-20))
        )
    else:
        pause_db = np.nan

    return {
        "f0_recall": f0_recall,
        "f0_gpe": f0_gpe,
        "f0_rmse_cents": f0_rmse,
        "vuv_acc": vuv,
        "ltass_rmse_db": ltass_rmse,
        "ltass_max_db": ltass_max,
        "fric_db": fric,
        "mod_corr": modulation_correlation(r16, e16),
        "pause_db": pause_db,
        "_ltass_dev": dev,  # per-band deviation (for plots / reports)
    }


# Direction of "better" for every scalar metric (used by the gate and reports).
HIGHER_IS_BETTER = {
    "si_sdr": True,
    "estoi": True,
    "pesq": True,
    "f0_recall": True,
    "vuv_acc": True,
    "mod_corr": True,
    "f0_gpe": False,
    "f0_rmse_cents": False,
    "ltass_rmse_db": False,
    "ltass_max_db": False,
    "fric_abs_db": False,
    "pause_db": False,
}
