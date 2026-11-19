# Plan: Battlefield-noise synthetic test bench + software fixes that affect hardware

## Context

The review found the current system can't be verified. There's no training, no evaluation and no data generator, and the demo metrics are misleading. Several software defects also hit the hardware directly: A/B desync, clock drift, level handling, clipping, and real-time threading.

Goals:
1. Learn what battlefield noise and human speech look like in frequency and level, and encode that in a synthetic generator for **both mic inputs**.
2. Test the pipeline on it, with the explicit success criterion that **the speech properties we synthesise survive to the output**.
3. Fix every software problem that affects the hardware, choosing the lowest-compute solution at each step **without lowering output quality**. Every change passes a measured non-regression gate.

User-confirmed facts:
- **Build:** two I2S MEMS mics wired **directly to the Raspberry Pi 5's I2S port** (no ESP32, no USB hop, no microcontroller firmware). The Pi captures through ALSA.
- **Geometry:** primary mic faces the lips; the reference mic sits directly behind it, facing away. The pod is about 5–6 cm from the lips.
- **Level:** 130 dB SPL is at the mic itself (worst case).
- **Speech data:** synthetic plus real speech are both allowed.
- **Training:** a GPU is available for fine-tuning.

### What the geometry changes (the key insight)
- Back-to-back mics with spacing d ≈ 1–2 cm form a **near-field endfire pair**.
  - Speech level difference between mics: 20·log10((r1+d)/r1) ≈ **1.5–3 dB**, plus a few dB of HF board shadow.
  - Inter-mic delay: d/c ≈ 30–60 µs ≈ 1.4–2.8 samples at 48 kHz.
- The current `RobustNLMS` speech gate needs a **12 dB** level difference, so it can never fire. In this geometry the NLMS will cancel speech.
- Noise coherence between mics stays high up to c/(2d) ≈ 8.5–11 kHz. Noise from the front reaches the primary first, which is non-causal for the current filter.
- The cheapest correct structure is therefore a **differential / GSC beamformer**: a front beam toward the mouth plus a noise reference with a null on the mouth. It costs a few to a few hundred MACs per sample, versus 512 for the current 256-tap per-sample Python NLMS.
- At 130 dB the INMP441 (AOP 120 dB) clips about 10 dB deep. Software can only detect, contain and quantify this. The bench will **quantify** how much a high-AOP mic would help, so the hardware decision rests on data.

---

## Knowledge base used by the generator
Values are literature-typical ranges. The generator **randomises within ranges** (domain randomisation), so no single number is critical. Step 1.0 cross-checks them against sources (e.g. MIL-STD-1474E, jet-noise similarity spectra, Peterson–Barney vowel data) before freezing.

### Noise sources
| Source | Dominant frequency content | Generator model | Level at mic |
|---|---|---|---|
| Jet (turbofan / fighter) | Broadband jet-mixing hump peaking 100 Hz–1 kHz. Fan blade-pass tone BPF = blades×RPM/60 ≈ 1–5 kHz plus harmonics. "Buzz-saw" shaft-order tones (k × 50–170 Hz). Turbine whine 3–10 kHz. Afterburner crackle | Shaped noise (peak + slopes) + slowly drifting tone stacks + skewed-impulse crackle | 115–150 dB |
| Helicopter | Main rotor BPF 10–30 Hz plus harmonics. Blade slap (impulsive, 5–20/s). Tail rotor 50–120 Hz. Turbine whine 3–8 kHz | Harmonic series + periodic impulses + whine | 100–125 dB |
| Tank / tracked vehicle | Track-pitch frequency v/p ≈ 10–100 Hz with many harmonics. Diesel firing frequency n_cyl·RPM/120 ≈ 50–260 Hz plus harmonics (or gas-turbine whine). Clank transients. Mostly below 1 kHz | Two harmonic stacks with speed/RPM drift + random clanks + LF broadband | 105–125 dB (interior) |
| Missile / rocket launch | Broadband rumble peaking 20–500 Hz (smaller motor gives a higher peak). Crackle (positively skewed pressure). Launch transient, lasting seconds | Shaped noise + crackle + onset envelope | 130–160 dB |
| Gunfire | Muzzle blast (Friedlander, 0.3–3 ms) with energy 0.2–3 kHz. Ballistic N-wave crack 1–5 kHz | Friedlander + N-wave pulses, Poisson / burst timing | 140–170 dB peak (always clips) |
| Artillery / explosion at distance | Friedlander, 5–50 ms, energy below 200 Hz, low-passed with distance | Pulse + distance LPF | 120–150 dB peak |
| Wind / rotor wash at the mic | Turbulence below 500 Hz, **incoherent between mics**, gusty (0.1–2 Hz) | Independent per-mic LF noise with AM | can saturate |
| Radio / babble (optional) | Competing speech | Real speech, far-field | 70–90 dB |

Spatial models:
- Far-field point sources at random azimuths: front, side and **behind**.
- Diffuse field with the theoretical two-mic coherence sinc(2πfd/c).
- Wind as per-mic independent noise.

### Human speech
| Property | Range | Why it matters |
|---|---|---|
| F0 | Male 85–155 Hz, female 165–255 Hz. Lombard effect raises it 10–50% | Overlaps tank/diesel harmonics; risk the model kills voicing |
| Formants | F1 250–900, F2 850–2500, F3 2000–3500 Hz (female about 15–20% higher) | Overlaps jet buzz-saw and BPF tones |
| Fricatives / bursts | /s/ 4–8+ kHz, /ʃ/ 2.5–6 kHz, /f θ/ weak broadband. Plosive bursts 10–30 ms | Intelligibility; first thing denoisers erase |
| Energy vs intelligibility | Energy concentrated 100–1000 Hz; intelligibility weight 1–4 kHz | Defines the bands we must protect |
| Temporal | Syllable modulation 3–6 Hz (peak about 4 Hz); 20–40% pauses | Basis of STOI-type metrics and VAD |
| Level at 5 cm | Shouting / Lombard 88–95 dB(A) at 1 m, about +26 dB at 5 cm, giving **114–121 dB SPL**. Normal raised speech 96–102 dB | Speech itself can approach the INMP441 AOP |

Consequences: mic SNR at 130 dB is about −10 to −30 dB broadband (better at 1–4 kHz for low-frequency-heavy noise). Harmonic noises (tank, diesel, jet buzz-saw) look like voiced speech, so the property tests must focus on them.

---

## Phase 0: Unblock (prerequisite, small)
Problem: nothing runs portably. Fixes:
- Remove `D:/...` hardcodes. Move `HighSPLPipeline` from `laboratory/dsp/pipeline.py` into the empty `src/highspl/pipeline.py` as the **single streaming pipeline** used by the lab UI, the evaluator and the Pi.
  - Config-driven via `configs/`. Reuse `src/highspl/config.py` `load_config` and update the stale `base.yaml`.
  - Add `reset()`.
- Make `FastEnhancerConfig` require only the file its backend uses (`src/highspl/models/fastenhancer.py:38`).
- Split `pyproject.toml` dependency groups:
  - `runtime`: numpy, scipy, soxr, onnxruntime, sounddevice
  - `lab`: PyQt6, pyqtgraph, customtkinter
  - `train`: torch and friends, keeping the cu126 index only for x86 via markers
  - `eval`: pesq, pystoi, librosa, pyroomacoustics
- Environment: `uv` with Python 3.11. The machine default is 3.14, which `requires-python` rejects.

## Phase 1: Synthetic dual-mic data generator
Fills existing empty stubs and adds new modules.

| File | Content |
|---|---|
| `src/highspl/data/speech_synth.py` (new) | Parametric source-filter voice: glottal pulse train with F0 contour, jitter/shimmer and declination; formant resonators from a vowel table; fricative and plosive noise bursts; syllabic envelope and pauses; Lombard mode (level, F0, tilt). **Emits labels**: F0 track, V/UV mask, phone segments, target formants |
| `src/highspl/data/noise_synth.py` (new) | One generator per row of the noise table, parameters randomised in range, seedable |
| `src/highspl/data/rir.py` | Geometry and propagation: near-field mouth point source at r1 ∈ [4,7] cm on axis; mic spacing d ∈ [0.8,3] cm (endfire, reference behind); exact fractional delays, 1/r spreading, simple HF board-shadow shelf (0–6 dB, randomised). Environments: free field / ground reflection, and vehicle cabin via `pyroomacoustics` (RT60 0.15–0.5 s) |
| `src/highspl/data/clipping.py` | **Hardware model** (see chain below) |
| `src/highspl/data/mixture.py` | Scene composer: 1–3 noise sources + diffuse + optional wind + impulses; SPL calibration in dB SPL at the mic; Lombard speech level tied to noise level; writes 2-channel mic input, **clean target** (speech direct path at primary mic, linear mic response, no noise or clipping), noise-only pair, labels JSON |
| `scripts/generate_mixtures.py` | CLI: frozen eval set `eval_v1` (seeded, about 240 scenes × 10 s, gitignored under `data/generated/`, manifest plus hash committed) and a tiny on-the-fly CI subset |

Hardware model chain, in order:
1. Sensitivity (−26 dBFS at 94 dB SPL, so 0 dBFS = 120 dB SPL), with ±1 dB per-mic mismatch
2. Roughly 60 Hz HPF
3. Overload: soft knee then hard clip at AOP. AOP is a parameter: 120 dB for INMP441, versus 130/135 dB for alternative mics
4. Self-noise (33 dBA ≈ −87 dBFS)
5. Roughly 15 kHz LPF
6. **I2S L/R sampling offset** of 0–1 sample (fractional)
7. 24-bit quantisation (I2S data is 24-bit in a 32-bit slot; ALSA hands the Pi 24-bit samples)
8. Optional software DC blocker (off by default: the mic's own ~60 Hz high-pass removes DC)
9. ALSA overrun: lost capture periods (both channels together, zero-filled)
10. Capture-vs-playback clock drift of ±50 ppm (for I/O tests)

Speech tiers:
- **S1** parametric, with exact ground truth.
- **S2** real speech: LibriSpeech test-clean (CC BY 4.0) plus DNS clips via the existing `scripts/download_dns_voice.py`. Speakers are split train/val/test.
- **S3** S2 with a Lombard transform (level plus tilt EQ).

Real-noise sanity check: UrbanSound8K `gun_shot` and `engine_idling` via `scripts/download_urbansound.py`, used only to compare spectra and modulation statistics against the synthetic generators.

**Challenge.** The pretrained FastEnhancer learned real speech, so it may treat S1 vowels (stationary harmonic stacks) as engine tones and erase them. That would be a false alarm about the system, not a real failure.
- **Resolution:** use S2/S3 for headline quality.
- Use S1 for exact property checks and as a stress test of harmonic-noise confusion.
- Report the tiers separately, never averaged.

## Phase 2: Evaluation harness and baseline
- `src/highspl/evaluation/speech_properties.py` (new): the **property-preservation scorecard**, comparing output to the clean target (and to known parameters for S1):
  - F0 gross pitch error and fine RMSE in cents (`librosa.pyin`)
  - V/UV accuracy
  - Formant error (LPC; S1)
  - LTASS deviation per 1/3-octave band (100 Hz–8 kHz)
  - Fricative band (4–8 kHz) energy preservation in unvoiced segments
  - Modulation-spectrum (2–8 Hz) correlation
  - Residual noise level in speech pauses
- Reuse `src/highspl/evaluation/metrics.py` (`calculate_si_sdr`, `calculate_snr`, `calculate_coherence`), plus `pystoi` (ESTOI) and `pesq` (WB at 16 kHz via `soxr`).
- `scripts/evaluate.py` runs each pipeline variant through the **same streaming path as real time** (512-sample blocks via `src/highspl/pipeline.py`).
  - It aligns output to target by cross-correlation, which also gives the **measured algorithmic latency**.
  - It records per-block time p50/p99/p99.9, RTF and peak RSS.
  - `src/highspl/evaluation/report.py` writes CSV plus a Markdown/HTML report, broken down by noise class × SPL band × speech tier.
- **Baseline B0** = the current system (ROBUST NLMS + FastEnhancer-B ONNX, ×0.8). Also BYPASS (AI only) and FDAF, for reference.

**Non-regression gate (enforces "don't decrease output").** Every later change is accepted only if, on `eval_v1` with paired bootstrap 95% CI:
- ΔESTOI ≥ −0.005
- ΔSI-SDR ≥ −0.2 dB
- ΔPESQ ≥ −0.05
- No property metric worse beyond its tolerance (F0 GPE +1 pt, LTASS ±0.5 dB, fricative band ±1 dB)

Compute savings are taken **only** under this gate. Absolute targets (e.g. F0 GPE ≤ 5%, LTASS within ±3 dB from 300 Hz to 4 kHz) are set after B0 is measured.

## Phase 3: Software problems that affect the hardware
Each item lists the problem, the challenge considered, the chosen solution, and its cost.

1. **A/B desync** (`laboratory/audio/capture.py:192`). Use one queue of **paired stereo frames** (single append per callback). For two-device lab mode, timestamp-pair the frames.
   - Considered: sync locks between two queues. Rejected because pairing at the source is simpler and free.
2. **Polling `sleep(0.01)` and silent underruns.** Use blocking `queue.Queue.get(timeout)` and add xrun counters. Lower CPU than today.
3. **Worker dies on exception or NaN** (`laboratory/ui/main_window.py:495`, `src/highspl/models/fastenhancer.py:241`). Per-block `try`; sanitise non-finite values and soft-reset only the failing stage; fall back to primary passed through a limiter; raise a health flag to the UI. Never raise in the audio path.
4. **Stale state and orphan threads.** Call `pipeline.reset()` on start, join the worker on stop, and reset a filter's state when its mode switches.
5. **Capture ↔ playback clock drift (±50 ppm).** Linear-interpolation fractional resampler on the output with a PI controller on queue depth.
   - Considered: drop or insert samples (cheapest, but clicks) and `soxr` (no variable ratio).
   - Chosen cost: about 2 MAC per sample.
6. **Level management** (live path has none; offline uses non-causal peak normalisation in `scripts/live_capture.py`):
   - Calibrated dBFS↔dB SPL mapping.
   - Slow, gain-bounded streaming **AGC before the neural network**, frozen during impulses.
   - Output target level plus a fast **hearing-safety limiter**, replacing the ×0.8 hack and upgrading `src/highspl/dsp/limiter.py` from per-block scaling to an attack/release envelope.
   - Cost: about 3–5 ops per sample.
7. **Clipping and impulses.** Per-channel clip-run detector; freeze adaptation during clipping; **channel diversity** (if one mic clips, take the other, gain-matched, for those samples); short impulse ducking with ramps (protects ears and avoids neural artifacts); clip-% telemetry.
   - Out of scope: declipping models, too costly.
8. **Mic mismatch and I2S L/R offset**, which matter now because the inter-mic delay is only about 2 samples. Run a background calibrator in noise-only periods: per-band gain match plus fractional-delay estimate (GCC-PHAT). Updates every few seconds, saved per device. Negligible cost.
9. **Pi I2S capture (replaces the ESP32/USB path; nothing to flash).**
   - Both mics share one I2S bus, so ALSA delivers interleaved stereo frames: A/B stay aligned by construction. The lab app's `shared_stereo` path already assumes this.
   - Capture through `sounddevice` on the I2S ALSA card (48 kHz, 2 ch, 24-bit data), with small periods (128–256 samples) to cut latency. This removes the ESP32's 10.7 ms packetisation and the USB hop.
   - Handle ALSA **overruns** (xruns): count them, zero-fill the lost period in both channels together, and tell the pipeline so it can reset adaptation history instead of learning from a gap.
   - Deployment doc: device-tree overlay / ALSA card setup for two I2S mics on the Pi 5, to be **verified on the real board** (cannot be tested from this repo).
   - The `ESP32` hardware doc is replaced by a Pi-direct I2S doc.
10. **Pi runtime:**
    - ORT `SessionOptions`: intra-op 1–2 threads, sequential, full graph optimisation, preallocated I/O.
    - `gc.freeze()` and GC disabled inside the loop; no prints in the audio path.
    - CPU affinity (I/O core vs compute core), `chrt` real-time priority, performance governor, active cooler, a 30-minute thermal soak.
11. **Honest telemetry.**
    - Replace the energy-ratio "noise reduction" figure with pause-noise level before/after (VAD-based).
    - Replace hardcoded latency with a loopback-measured value.
    - Show xruns, clip %, adaptation-frozen %, and p99 block time in the lab UI.
12. **Latency truth.** Document the **measured** algorithmic latency (neural stage: 512 samples = 10.7 ms, measured by the evaluator) plus the capture and playback periods. Trim the ALSA periods and verify by loopback. Fix the README claims.

## Phase 4: DSP front-end redesign (ablation, geometry-aware)
All candidates are block-vectorised NumPy on the existing 512-sample buffer, so they add **no extra latency**.
- D0 bypass
- D1 current RobustNLMS (B0)
- D2 FDAF
- **D3** fixed endfire differential beamformer: front hypercardioid with near-field correction, bounded low-frequency EQ (≤ +12 dB), crossover to primary-only above c/(2d). About 10–20 MAC per sample.
- **D4** adaptive first-order differential (Elko-style single β), null steered over the rear half-plane. About 20 MAC per sample.
- **D5** robust GSC:
  - D3 as the fixed beam.
  - Adaptive blocking filter (16–32 taps; adapts only when speech dominates, judged by front/back beam power ratio) to null the mouth.
  - Noise canceller: FDAF with B=128, P=1–2, per-bin coherence and level-difference gating; adapts only when noise dominates; primary delayed a few samples for non-causal noise.
  - About 150–300 MAC per sample.
- Selection rule: the cheapest variant whose (variant + neural network) result passes the gate against the best variant.
- Expected outcome, to be verified rather than assumed: D5 or D4 replaces the per-sample Python NLMS at a fraction of the compute.
- If the per-sample NLMS family is kept anywhere, fix:
  - FDAF partition normalisation
  - the gain-mismatch deadlock (use the calibrated gain)
  - the int16 heuristic
  - the VSS level-dependence
  - the per-sample divergence gate
- **Challenge.** Differential arrays boost low-frequency self-noise and are fragile to mismatch. Mitigations:
  - bounded EQ
  - calibration (Phase 3, item 8)
  - fading beamformer strength with noise level (no low-frequency boost when quiet)

## Phase 5: Neural stage (GPU available)
- **N1** AGC in front of FastEnhancer-B (from Phase 3, item 6). Test first: likely a large win at near-zero compute.
- **N2** ORT tuning (compute only, gated).
- **N3** Fine-tune FastEnhancer-B from `00500.pth` on **dynamic-mixed** generator scenes.
  - Mainly S2/S3 plus some S1; clipping, impulse and Lombard augmentation.
  - Input is the **output of the chosen Phase 4 front-end**, so the model learns that residual distribution.
  - Use the upstream loss recipe (see `models/fastenhancer/b/config.yaml`: mag_mse 0.3, complex_mse 0.2, consistency 0.3, wav_l1 0.2), optionally plus SI-SNR.
  - Use `src/highspl/training/losses.py` only after fixing its ERLE weighting (smoothed VAD, not instantaneous power) and dropping the output-clipping term.
  - Same architecture, so **zero runtime cost increase**.
  - Uses the upstream `D:\SIH\fastenhancer` training code; implement `src/highspl/training/trainer.py` / `scripts/train.py` as a thin wrapper.
- **N4** Dual-channel variant: 2-channel input (D3 beam plus blocking output, or raw mics) by widening only the first convolution, costing under a few percent more compute. Adopt only if it beats D5+N3 beyond the gate margin.
- **N5** Compute reductions *after* quality is up: smaller FastEnhancer variant, then ORT INT8 quantisation. Each is accepted only if it passes the gate.
- **N6** Streaming ONNX export plus parity tests (reuse `tests/integration/test_fastenhancer_parity.py` patterns).

## Phase 6: On-device validation (Pi 5 + mics)
- Run `eval_v1` through the Pi pipeline offline (bit-exact path) and confirm the metrics match the PC.
- Measure real-time p99.9, RTF and thermal behaviour over 30 minutes. Update `scripts/benchmark_pi5.py` to report and gate on p99.9 rather than the mean.
- Live: loudspeaker noise scenes plus a small speaker as an artificial mouth at 5 cm. Real 130 dB can't be reproduced safely, so this validates the chain and calibration, not the extremes.
- Measure glass-to-glass latency by loopback.

## Phase 7: Hardware decision report (from the simulation)
Using the same bench, run these sweeps:
- **AOP:** 120 (INMP441) vs 130 vs 135 dB. Candidates such as high-AOP I2S/PDM MEMS parts; check datasheets and that the Pi 5's I2S/PDM interface supports them.
- Mic spacing d from 0.8 to 3 cm.
- Pod distance r1 from 4 to 7 cm.
- Board shadow.

Output: quality vs SPL curves, the SPL where each option breaks, and a recommended mic, spacing and placement.

---

## Ordering
- Phase 0 → 1 → 2 (baseline B0).
- Then Phase 3, Phase 4 and N1/N2 in parallel.
- Then N3/N4 on the GPU.
- Then N5.
- Then Phase 6.
- Phase 7 can start as soon as Phase 2 exists.
- Every step reports its gate result plus compute delta in the eval report.

## Critical files
- Pipeline: `src/highspl/pipeline.py`, `laboratory/dsp/pipeline.py` (becomes a thin wrapper), `laboratory/audio/capture.py`, `laboratory/ui/main_window.py`
- Data: `src/highspl/data/{speech_synth,noise_synth,rir,clipping,mixture}.py`, `scripts/generate_mixtures.py`
- Evaluation: `src/highspl/evaluation/{speech_properties,report}.py`, `scripts/evaluate.py`
- DSP: `src/highspl/dsp/{beamformer,agc,calibration,drift}.py` (new), `limiter.py`, `fdaf.py`, `nlms.py`
- Models / training: `src/highspl/models/fastenhancer.py`, `src/highspl/training/losses.py`, `src/highspl/training/trainer.py`, `scripts/train.py`
- I/O: `laboratory/audio/capture.py` (paired stereo frames, xrun handling, drift control); Pi I2S setup doc
- Config and CI: `configs/base.yaml`, `configs/data/*.yaml`, `pyproject.toml`, `.github/workflows/ci.yml`

## Verification
- Unit tests (`uv run pytest`):
  - generators produce the requested F0, formants, band levels and SPL
  - hardware model clips at AOP and injects the configured offset and mismatch
  - beamformer nulls an on-axis near-field source to the expected depth
  - drift controller holds queue depth
  - paired queue never desyncs under a randomised producer/consumer test
  - the empty `test_mixture.py` and `test_pipeline.py` get filled
- `uv run python scripts/generate_mixtures.py --set eval_v1`, then `uv run python scripts/evaluate.py --set eval_v1 --pipelines B0,D3,D4,D5,...`. Inspect the report; the gates must pass.
- CI runs the tiny on-the-fly synthetic subset, with the ONNX model fetched from a release asset, so the AI path is actually exercised.
- On the Pi: updated `benchmark_pi5.py` (p99.9 against the 10.67 ms deadline), the 30-minute soak, and the loopback latency test.

---

## Implementation status (end of this pass)

Evidence for every "done" below is in `docs/EVAL_REPORT.md`; 259 tests pass, ruff is clean.

| phase / item | status | notes |
|---|---|---|
| 0 Unblock (shared pipeline, config, deps split, no `D:/` paths) | **done** | `src/highspl/pipeline.py`; presets in `configs/pipelines/`; runtime / lab / eval / train dependency groups |
| 1 Synthetic dual-mic generator (speech S1/S2/S3, 7 noise classes, pod geometry, hardware model, frozen `eval_v1`) | **done** | vehicle-cabin (`pyroomacoustics`) environment **not** generated |
| 2 Evaluation harness, speech-property scorecard, baseline B0, non-regression gate | **done** | gate corrected during the run (per-scene SI-SDR clip, operating-range verdict) |
| 3.1 A/B pairing, 3.2 blocking queues, 3.3 contained worker, 3.4 reset + join | **done** | `laboratory/audio/capture.py`, `laboratory/ui/main_window.py`, `scripts/live_call_ui.py` rebuilt on the shared classes |
| 3.5 clock-drift compensation | **done** | PI-controlled resampler; held within one block at +/-50..200 ppm in simulation |
| 3.6 level management (AGC, lookahead limiter) | **done** | AGC target tuned on the bench (-12 dBFS) |
| 3.7 clipping / impulses | **partial** | adaptation freeze on clipping, AGC impulse hold, limiter. Channel diversity **not** implemented (both capsules clip on the same samples); separate impulse ducking not implemented |
| 3.8 mic mismatch / I2S offset calibration | **done** | estimator accurate (gain within ~3 %); +1.2 dB SI-SDR for the un-fine-tuned GSC; not combined with the fine-tuned model |
| 3.9 Pi I2S capture | **partial** | shared-stereo capture path + bring-up doc; **not run on a real Pi** (no ESP32 in this build, so the ESP32 transport work was dropped) |
| 3.10 Pi runtime | **partial** | ORT single-thread sequential session; `gc.freeze`, CPU affinity and RT priority are **not** implemented |
| 3.11 honest telemetry, 3.12 latency truth | **done** | noise-floor tracker, latency lower bound, measured algorithmic latency |
| 4 front-end ablation (D3 diff, D4 adiff, D5 GSC, noise-level blend) | **done** | GSC + blend selected; legacy NLMS/FDAF kept unchanged as baselines (their bugs were not fixed: they lose) |
| 5 N1 AGC, N2 ORT options, N3 fine-tune, N6 export parity | **done** | validation SI-SDR -1.68 -> +4.08 dB; gate ACCEPTED |
| 5 N4 two-channel neural model, N5 smaller model / INT8 | **not done** | N5 unnecessary on the PC numbers; to be revisited after a Pi benchmark |
| 6 On-device validation | **not possible here** | `scripts/benchmark_pi5.py` rewritten (p99.9 gate); run only on the PC |
| 7 Hardware decision report | **done** | AOP, spacing and pod-distance sweeps (indicative, 24 scenes per variant) |

### Outstanding risks, in order of importance

1. **Everything is simulated.** The noise models are literature-typical, not recorded, and the fine-tuned model has only seen this generator. Validate on real recordings before relying on any number.
2. **The Pi 5 is unmeasured** (latency tail, thermals, I2S/ALSA behaviour).
3. **The microphone's 120 dB overload point** remains the ceiling at 130 dB SPL; the sweep suggests a 130-135 dB-AOP part gives the biggest hardware gain.
4. At the quietest SPL band the fine-tuned system has lower SI-SDR than AI-only while intelligibility, PESQ, pitch and rhythm improve.
