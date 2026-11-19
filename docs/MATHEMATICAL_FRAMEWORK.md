# Mathematical framework

What the code in `src/highspl` actually computes. Every statement here is
implemented and tested; numbers come from `docs/EVAL_REPORT.md`.

## 1. Signals and geometry

Two omnidirectional MEMS capsules form a **back-to-back, near-field endfire
pair**: the primary mic `f` faces the lips, the reference `r` sits directly
behind it facing away, at spacing `d` (1-2 cm), with the lips at `r1` (4.5-6.5 cm).
With `tau = d/c` (about 2 samples at 48 kHz):

| source | model at the rear mic |
|---|---|
| far-field wave from the front | `r(t) = f(t - tau)` |
| far-field wave from the rear | `r(t) = f(t + tau)` |
| mouth (spherical wave) | `r(t) = g f(t - tau)`, `g = r1/(r1+d) ~ 0.79` |
| diffuse field | coherence `sinc(2 pi f d / c)` |
| wind | independent per capsule (zero coherence) |

The mouth differs from a front far-field wave only by the amplitude factor `g`
(about -2 dB). This is the key constraint of the design: there is almost no
spatial discrimination for noise arriving from the front.

> The earlier model, `d(n) = s + n_p`, `r(n) = n_r` with `r` "noise only", does
> not hold for this geometry: the reference hears the talker at -2 dB, so any
> filter that treats `r` as a noise reference learns to cancel speech.

## 2. Stage 1 (selectable; `pipeline.frontend.mode`)

All are block-vectorised and add no look-ahead.

**diff (D3)** - first-order differential beamformer. With the two cardioids
`c_F = f(t) - r(t - tau)` (null behind) and `c_B = r(t) - f(t - tau)`:

```
y = EQ{ c_F - beta c_B }
```

`EQ` flattens the *mouth* response `M(w) = 1 - g e^{-j2w tau} + beta (1-g) e^{-j w tau}`
(it is only -13 dB at low frequencies) with three shelving biquads fitted by
least squares, each bounded by +12 dB so low-frequency self-noise is not boosted
without limit. Fractional delays use a 19-tap Kaiser-sinc FIR; both paths are
delayed by 8 samples so the filters are causal.

**adiff (D4)** - `beta` adapted per 128-sample sub-block to null the strongest
rear-half-plane noise: `beta = clip(<c_F,c_B> / <c_B,c_B>, 0, beta_max)` (Elko).

**gsc (D5)** - generalised sidelobe canceller:

```
b(t) = D3 output (fixed beam)
u(t) = r(t) - g f(t - tau)            # blocks the mouth exactly: u = 0 for speech
y(t) = b(t - L/2) - sum_k w_k u(t - k)  # w: constrained block-NLMS, L = 32
```

`w` adapts only when speech is absent (`P_b/P_u` stays near its running minimum),
never on a clipped block, with `|w|_1 <= 6` and leakage. The blocking gain `g`
must match the real mouth path: a 1 dB capsule mismatch leaks about 9 % of the
speech into `u`, which `w` (about 4.7 for front noise) would then subtract.
`highspl.dsp.calibration` therefore estimates `(g_eff, tau_eff)` from clean speech
by the cross-spectrum `H = csd(f,r)/psd(f)` (gain `|H|`, delay from the phase
slope) and corrects the rear channel.

**blend** (`frontend.blend`) - cross-fades from the primary mic to the beamformer
by the noise floor of the primary (10th percentile of block power over 4 s):
the beamformer costs a little speech fidelity when there is nothing to remove.

**robust / vss / nlms / fdaf** - the legacy filters, kept as baselines. Their
control logic assumes a reference that does not contain speech; see section 1.

## 3. Level management

* **AGC** (before the neural stage): block RMS with fast attack / slow release,
  gain slew-limited (`40 dB/s` down, `6 dB/s` up), held through impulses
  (crest factor > 18 dB) so bangs do not pump it. The stock enhancer was trained on
  mixtures at about -25..-10 dBFS. On the bench a target of -24 dBFS made it
  suppress speech in loud scenes; -12 dBFS was the best of -12/-16/-20 and is the
  default of the recommended preset (`ft_gsc_blend_agc12`).
* **Lookahead limiter** (after it): with `req[n] = min(1, ceiling/|x[n]|)`,
  `T[i] = min(req over the trailing hold+S window)` and
  `G[m] = mean(T[m..m+S-1])`, the output satisfies `|y| <= ceiling` exactly while
  the gain ramps down *before* the peak (S = 48 samples).

## 4. Stage 2: neural enhancement

FastEnhancer-B (causal RNNFormer, 48 kHz STFT 1024/512 with power-law
compression 0.3) runs through ONNX Runtime one 512-sample hop at a time. Its
measured algorithmic delay is **512 samples (10.7 ms)**. The pipeline never sees
a model exception: a failing stage is reset and the block falls back to a safe
signal (`telemetry["healthy"] = False`).

### Fine-tuning loss (`highspl.training.trainer`)

The upstream training recipe, on the compressed spectra `S = |X|^0.3 e^{j arg X}`:

```
L = 0.3 MSE(|S_hat|, |S|) + 0.2 MSE(S_hat, S) + 0.3 MSE(STFT512(y_hat), STFT512(y)) + 0.2 |y_hat - y|_1
```

The earlier custom composite loss (`training/losses.py`) is retained but its two
unsound terms were corrected: the ERLE "speech absent" weight is computed from a
10 ms envelope rather than instantaneous sample power, and the output-clipping
penalty is off by default (clipping happens at the input, where an output
penalty cannot act).

## 5. Metrics (`highspl.evaluation`)

All level-invariant, computed on the output aligned to the clean target:

* SI-SDR, ESTOI, PESQ-WB (quality)
* speech **property preservation**: F0 recall and gross error, voicing accuracy,
  long-term-spectrum deviation (1/3-octave), 4-8 kHz fricative energy, 2-8 Hz
  syllabic-envelope correlation, residual level in pauses

"Noise reduction" is not reported as `10 log10(E_out/E_in)`: that counts speech
as noise and scores a muted output as perfect. The live UI reports the noise
floor (minimum statistics) before and after instead.
