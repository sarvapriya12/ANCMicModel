# Evaluation report

All numbers below were measured on the **synthetic dual-microphone bench** (`docs/SYNTHETIC_DATA.md`), through the real streaming pipeline, block by block. Nothing here was measured on real hardware or real battlefield recordings. Regenerate with the commands at the end.

## 1. Decisions this report supports

| question | answer | evidence |
|---|---|---|
| Is the legacy pipeline (B0: hard-gated NLMS + FastEnhancer, x0.8) usable? | **No.** It scores far below the raw microphone at every loudness | section 3 |
| Does a geometry-aware dual-mic front-end help? | **Yes**, +0.07 ESTOI over AI-only, most between 105 and 130 dB | sections 3-4 |
| Is the per-sample NLMS worth keeping? | **No**: it is the slowest stage and the only one that cancels speech | sections 3, 6 |
| Does fine-tuning the neural stage on the cascade help? | **Yes**, large: +0.16 ESTOI, +2.2 dB SI-SDR over AI-only; accepted by the gate | sections 4-5 |
| Is the extra compute justified? | The final pipeline is **cheaper than B0** (about 58 % of its block time on the PC) and far better | section 6 |
| Can software fix 130 dB SPL? | **Partly.** The fine-tuned system recovers real intelligibility at 120-138 dB, but the microphone's 120 dB overload point remains the ceiling | sections 4, 7 |

## 2. Method

* **Scenes:** `eval_v1`, 240 x 10 s: 6 noise groups (jet, helicopter, tank, missile, gunfire, artillery) x 4 SPL bands at the primary mic (90-105, 105-120, 120-130, 130-138 dB) x 3 speech tiers (S1 parametric voice 96 scenes, S2 real LibriSpeech test-clean 72, S3 the same with a Lombard transform 72). Speakers are disjoint from the training speakers (dev-clean).
* **Pod:** back-to-back pair, spacing 1-2 cm, lips at 4.5-6.5 cm, 0-1 sample I2S offset, +/-1 dB capsule mismatch, INMP441-class overload point (120 dB SPL), 24-bit capture.
* **Headline = S2 + S3** (real speech). S1 is reported separately because the pretrained model may mistake stationary synthetic vowels for engine tones; tiers are never averaged together.
* **Metrics** (all level-invariant, output aligned to the clean target): SI-SDR, ESTOI, PESQ-WB, and the speech-property scorecard (`highspl.evaluation.speech_properties`): F0 recall / gross error, voicing accuracy, long-term-spectrum deviation, 4-8 kHz fricative energy, 2-8 Hz syllabic-rhythm correlation, level in pauses.
* **AGC fairness:** for pipelines with an AGC the clean target is scaled by the same gain curve, so deliberate level normalisation is not scored as distortion.
* **Gate:** a candidate is accepted against a baseline only if, by paired bootstrap over scenes, the 95 % lower bound of every metric's improvement stays above its tolerance. Per-scene SI-SDR is clipped to [-15, 30] dB. The verdict uses the operating range (< 130 dB at the mic); the top band is reported for information. (The gate was first run unclipped over all bands and rejected every candidate purely because a handful of hopeless scenes at -25..-30 dB dominated the SI-SDR interval: a flaw of the gate definition, corrected before the final run.)

## 3. Pipelines compared

| preset | stage 1 | level | neural stage |
|---|---|---|---|
| raw | none (primary mic as captured) | - | - |
| **B0** | hard-gated NLMS (legacy) | x0.8 | stock FastEnhancer-B |
| **bypass** | none | x0.8 | stock (AI only) |
| gsc_blend | GSC, faded in with the noise level | x0.8 | stock |
| gsc_blend_agc12 | as above | AGC -12 dBFS + limiter | stock |
| adiff_blend | adaptive differential, faded in | x0.8 | stock |
| gsc_blend_cal | GSC + one-off quiet-room calibration | x0.8 | stock |
| **ft_gsc_blend_agc12** | GSC, faded in | AGC -12 dBFS + limiter | **fine-tuned** |

## 4. Results

### Real speech (S2 + S3, 144 scenes), all SPL bands

| pipeline | n | si_sdr | estoi | pesq | f0_recall | f0_gpe | ltass_rmse_db | fric_abs_db | mod_corr | pause_db |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 144.000 | -2.810 | 0.389 | 1.150 | 0.390 | 0.354 | 2.195 | 2.859 | 0.492 | -4.097 |
| B0 | 144.000 | -20.552 | 0.120 | 1.178 | 0.113 | 0.455 | 8.541 | 16.600 | 0.266 | -34.344 |
| bypass | 144.000 | -3.777 | 0.408 | 1.268 | 0.539 | 0.341 | 5.965 | 7.300 | 0.441 | -27.954 |
| gsc_blend | 144.000 | -4.575 | 0.481 | 1.380 | 0.583 | 0.247 | 5.831 | 8.776 | 0.523 | -37.598 |
| gsc_blend_agc12 | 144.000 | -3.548 | 0.501 | 1.435 | 0.576 | 0.208 | 4.327 | 8.447 | 0.572 | -43.706 |
| adiff_blend | 144.000 | -4.001 | 0.477 | 1.381 | 0.584 | 0.246 | 5.211 | 7.447 | 0.529 | -36.535 |
| gsc_blend_cal | 144.000 | -4.384 | 0.481 | 1.421 | 0.571 | 0.273 | 4.910 | 6.810 | 0.521 | -37.397 |
| ft_gsc_blend_agc12 | 144.000 | 2.164 | 0.579 | 1.540 | 0.643 | 0.192 | 2.635 | 3.380 | 0.704 | -42.168 |


### Real speech by SPL band at the primary mic

(0 = 90-105, 1 = 105-120, 2 = 120-130, 3 = 130-138 dB SPL)


**Band 0**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr | pause_db |
|---|---|---|---|---|---|---|---|
| raw | 36.000 | 9.473 | 0.697 | 1.354 | 0.773 | 0.763 | -10.688 |
| B0 | 36.000 | -18.773 | 0.146 | 1.151 | 0.107 | 0.373 | -47.023 |
| bypass | 36.000 | 13.999 | 0.805 | 1.684 | 0.866 | 0.791 | -65.001 |
| gsc_blend_cal | 36.000 | 13.449 | 0.825 | 1.805 | 0.880 | 0.793 | -66.948 |
| ft_gsc_blend_agc12 | 36.000 | 9.155 | 0.832 | 2.060 | 0.882 | 0.848 | -49.133 |


**Band 1**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr | pause_db |
|---|---|---|---|---|---|---|---|
| raw | 36.000 | 1.657 | 0.507 | 1.125 | 0.541 | 0.635 | -4.460 |
| B0 | 36.000 | -18.398 | 0.158 | 1.130 | 0.083 | 0.328 | -43.745 |
| bypass | 36.000 | 4.978 | 0.581 | 1.251 | 0.747 | 0.637 | -32.533 |
| gsc_blend_cal | 36.000 | 6.353 | 0.710 | 1.499 | 0.825 | 0.734 | -46.439 |
| ft_gsc_blend_agc12 | 36.000 | 6.574 | 0.769 | 1.750 | 0.839 | 0.826 | -48.058 |


**Band 2**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr | pause_db |
|---|---|---|---|---|---|---|---|
| raw | 36.000 | -7.108 | 0.260 | 1.050 | 0.184 | 0.409 | -0.738 |
| B0 | 36.000 | -19.435 | 0.116 | 1.244 | 0.111 | 0.250 | -35.942 |
| bypass | 36.000 | -10.121 | 0.198 | 1.082 | 0.369 | 0.285 | -9.472 |
| gsc_blend_cal | 36.000 | -10.043 | 0.297 | 1.122 | 0.417 | 0.409 | -24.382 |
| ft_gsc_blend_agc12 | 36.000 | 0.898 | 0.467 | 1.213 | 0.554 | 0.687 | -43.520 |


**Band 3**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr | pause_db |
|---|---|---|---|---|---|---|---|
| raw | 36.000 | -15.262 | 0.094 | 1.071 | 0.064 | 0.162 | -0.089 |
| B0 | 36.000 | -25.605 | 0.058 | 1.186 | 0.153 | 0.111 | -7.688 |
| bypass | 36.000 | -23.966 | 0.049 | 1.055 | 0.174 | 0.049 | -2.068 |
| gsc_blend_cal | 36.000 | -27.294 | 0.093 | 1.259 | 0.163 | 0.148 | -8.642 |
| ft_gsc_blend_agc12 | 36.000 | -7.970 | 0.250 | 1.138 | 0.298 | 0.455 | -27.370 |


### Real speech by noise type (operating range, bands 0-2)


**jet**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr |
|---|---|---|---|---|---|---|
| raw | 18.000 | -0.188 | 0.394 | 1.087 | 0.458 | 0.528 |
| bypass | 18.000 | -0.328 | 0.420 | 1.259 | 0.635 | 0.515 |
| ft_gsc_blend_agc12 | 18.000 | 4.014 | 0.655 | 1.596 | 0.766 | 0.762 |


**helicopter**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr |
|---|---|---|---|---|---|---|
| raw | 18.000 | 1.199 | 0.525 | 1.162 | 0.514 | 0.625 |
| bypass | 18.000 | 4.158 | 0.585 | 1.338 | 0.713 | 0.607 |
| ft_gsc_blend_agc12 | 18.000 | 5.147 | 0.682 | 1.558 | 0.754 | 0.761 |


**tank**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr |
|---|---|---|---|---|---|---|
| raw | 18.000 | 1.205 | 0.449 | 1.205 | 0.494 | 0.601 |
| bypass | 18.000 | 1.098 | 0.475 | 1.343 | 0.614 | 0.491 |
| ft_gsc_blend_agc12 | 18.000 | 5.785 | 0.690 | 1.738 | 0.781 | 0.837 |


**missile**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr |
|---|---|---|---|---|---|---|
| raw | 18.000 | 0.674 | 0.487 | 1.201 | 0.469 | 0.596 |
| bypass | 18.000 | 2.035 | 0.506 | 1.353 | 0.609 | 0.542 |
| ft_gsc_blend_agc12 | 18.000 | 3.930 | 0.624 | 1.514 | 0.681 | 0.736 |


**gunfire**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr |
|---|---|---|---|---|---|---|
| raw | 18.000 | 1.943 | 0.518 | 1.199 | 0.511 | 0.629 |
| bypass | 18.000 | 4.237 | 0.562 | 1.349 | 0.680 | 0.643 |
| ft_gsc_blend_agc12 | 18.000 | 7.366 | 0.738 | 1.827 | 0.793 | 0.825 |


**artillery**

| pipeline | n | si_sdr | estoi | pesq | f0_recall | mod_corr |
|---|---|---|---|---|---|---|
| raw | 18.000 | 3.211 | 0.552 | 1.205 | 0.550 | 0.634 |
| bypass | 18.000 | 6.513 | 0.622 | 1.391 | 0.711 | 0.629 |
| ft_gsc_blend_agc12 | 18.000 | 7.012 | 0.746 | 1.814 | 0.775 | 0.802 |


### S1 parametric voice (exact ground truth), all bands

| pipeline | n | si_sdr | estoi | pesq | f0_recall | f0_gpe | ltass_rmse_db | fric_abs_db | mod_corr | pause_db |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 96.000 | -4.378 | 0.347 | 1.062 | 0.356 | 0.475 | 3.631 | 2.210 | 0.470 | -3.600 |
| B0 | 96.000 | -19.777 | 0.173 | 1.143 | 0.119 | 0.513 | 8.962 | 18.684 | 0.301 | -26.260 |
| bypass | 96.000 | -7.821 | 0.394 | 1.240 | 0.523 | 0.389 | 6.464 | 6.165 | 0.463 | -17.066 |
| gsc_blend | 96.000 | -7.133 | 0.458 | 1.412 | 0.591 | 0.299 | 6.129 | 7.072 | 0.561 | -24.153 |
| gsc_blend_agc12 | 96.000 | -4.922 | 0.489 | 1.624 | 0.599 | 0.272 | 5.001 | 8.530 | 0.625 | -27.436 |
| adiff_blend | 96.000 | -6.981 | 0.444 | 1.366 | 0.579 | 0.308 | 5.839 | 6.613 | 0.558 | -23.419 |
| gsc_blend_cal | 96.000 | -6.459 | 0.468 | 1.435 | 0.576 | 0.322 | 5.713 | 6.304 | 0.562 | -23.253 |
| ft_gsc_blend_agc12 | 96.000 | 3.455 | 0.608 | 1.726 | 0.691 | 0.201 | 3.266 | 3.034 | 0.795 | -41.802 |


Reading the property columns: `f0_recall` is the share of voiced frames whose pitch survives within 20 %; `f0_gpe` is gross pitch errors among frames voiced in both; `mod_corr` is the 2-8 Hz syllabic-rhythm correlation; `pause_db` is the residual level in pauses relative to the active-speech level (lower is better). `ltass_rmse_db` and `fric_abs_db` are spectral-fidelity errors (lower is better); the raw microphone scores low on these only because its speech-active frames are speech-dominated, not because it sounds better.

### Non-regression gate vs AI-only (`bypass`), headline tiers, operating range


**B0: REJECTED** (all bands incl. 130-138 dB: would not pass)

| metric | delta (+=better) | 95% CI low | 95% CI high | tolerance | result |
|---|---|---|---|---|---|
| estoi | -0.3880 | -0.4391 | -0.3335 | -0.0050 | FAIL |
| si_sdr | -17.8342 | -19.7757 | -15.6579 | -0.2000 | FAIL |
| pesq | -0.1636 | -0.2534 | -0.0661 | -0.0500 | FAIL |
| f0_recall | -0.5601 | -0.6082 | -0.5112 | -0.0100 | FAIL |
| f0_gpe | -0.1680 | -0.2038 | -0.1344 | -0.0100 | FAIL |
| ltass_rmse_db | -3.3006 | -4.1251 | -2.4130 | -0.5000 | FAIL |
| fric_abs_db | -10.9687 | -12.5134 | -9.4577 | -1.0000 | FAIL |
| mod_corr | -0.2540 | -0.3028 | -0.2025 | -0.0100 | FAIL |


**gsc_blend: REJECTED** (all bands incl. 130-138 dB: would not pass)

| metric | delta (+=better) | 95% CI low | 95% CI high | tolerance | result |
|---|---|---|---|---|---|
| estoi | 0.0744 | 0.0505 | 0.1008 | -0.0050 | PASS |
| si_sdr | -0.3350 | -1.1655 | 0.4999 | -0.2000 | FAIL |
| pesq | 0.1208 | 0.0673 | 0.1764 | -0.0500 | PASS |
| f0_recall | 0.0526 | 0.0251 | 0.0811 | -0.0100 | PASS |
| f0_gpe | 0.0659 | 0.0356 | 0.0982 | -0.0100 | PASS |
| ltass_rmse_db | 0.4107 | -0.2304 | 1.0962 | -0.5000 | PASS |
| fric_abs_db | -1.2397 | -2.6431 | 0.1204 | -1.0000 | FAIL |
| mod_corr | 0.0632 | 0.0376 | 0.0894 | -0.0100 | PASS |


**gsc_blend_agc12: REJECTED** (all bands incl. 130-138 dB: would not pass)

| metric | delta (+=better) | 95% CI low | 95% CI high | tolerance | result |
|---|---|---|---|---|---|
| estoi | 0.0948 | 0.0666 | 0.1243 | -0.0050 | PASS |
| si_sdr | -0.4271 | -1.5257 | 0.7068 | -0.2000 | FAIL |
| pesq | 0.1191 | 0.0494 | 0.2079 | -0.0500 | PASS |
| f0_recall | 0.0490 | 0.0191 | 0.0817 | -0.0100 | PASS |
| f0_gpe | 0.1043 | 0.0686 | 0.1407 | -0.0100 | PASS |
| ltass_rmse_db | 1.5865 | 0.8402 | 2.3706 | -0.5000 | PASS |
| fric_abs_db | 0.5348 | -0.9436 | 1.9040 | -1.0000 | PASS |
| mod_corr | 0.1036 | 0.0730 | 0.1369 | -0.0100 | PASS |


**adiff_blend: REJECTED** (all bands incl. 130-138 dB: would not pass)

| metric | delta (+=better) | 95% CI low | 95% CI high | tolerance | result |
|---|---|---|---|---|---|
| estoi | 0.0677 | 0.0449 | 0.0935 | -0.0050 | PASS |
| si_sdr | -0.2367 | -0.9894 | 0.5725 | -0.2000 | FAIL |
| pesq | 0.1158 | 0.0651 | 0.1707 | -0.0500 | PASS |
| f0_recall | 0.0469 | 0.0207 | 0.0750 | -0.0100 | PASS |
| f0_gpe | 0.0641 | 0.0355 | 0.0950 | -0.0100 | PASS |
| ltass_rmse_db | 0.7576 | 0.1952 | 1.3575 | -0.5000 | PASS |
| fric_abs_db | -0.0485 | -1.2341 | 1.1660 | -1.0000 | FAIL |
| mod_corr | 0.0661 | 0.0427 | 0.0920 | -0.0100 | PASS |


**gsc_blend_cal: ACCEPTED** (all bands incl. 130-138 dB: would also pass)

| metric | delta (+=better) | 95% CI low | 95% CI high | tolerance | result |
|---|---|---|---|---|---|
| estoi | 0.0827 | 0.0569 | 0.1106 | -0.0050 | PASS |
| si_sdr | 0.8314 | -0.0186 | 1.7203 | -0.2000 | PASS |
| pesq | 0.1369 | 0.0801 | 0.1968 | -0.0500 | PASS |
| f0_recall | 0.0471 | 0.0180 | 0.0792 | -0.0100 | PASS |
| f0_gpe | 0.0639 | 0.0324 | 0.0975 | -0.0100 | PASS |
| ltass_rmse_db | 1.4178 | 0.7334 | 2.1095 | -0.5000 | PASS |
| fric_abs_db | 0.7015 | -0.4006 | 1.8784 | -1.0000 | PASS |
| mod_corr | 0.0745 | 0.0459 | 0.1046 | -0.0100 | PASS |


**ft_gsc_blend_agc12: ACCEPTED** (all bands incl. 130-138 dB: would also pass)

| metric | delta (+=better) | 95% CI low | 95% CI high | tolerance | result |
|---|---|---|---|---|---|
| estoi | 0.1611 | 0.1313 | 0.1925 | -0.0050 | PASS |
| si_sdr | 2.2205 | 0.8100 | 3.6900 | -0.2000 | PASS |
| pesq | 0.3356 | 0.2584 | 0.4088 | -0.0500 | PASS |
| f0_recall | 0.0981 | 0.0674 | 0.1324 | -0.0100 | PASS |
| f0_gpe | 0.1222 | 0.0882 | 0.1579 | -0.0100 | PASS |
| ltass_rmse_db | 2.2447 | 1.3593 | 3.1688 | -0.5000 | PASS |
| fric_abs_db | 2.2732 | 0.9142 | 3.7455 | -1.0000 | PASS |
| mod_corr | 0.2161 | 0.1758 | 0.2579 | -0.0100 | PASS |


## 5. Fine-tuning the neural stage

* Same architecture (FastEnhancer-B), so the ONNX export, size and compute are unchanged; the export was checked against the PyTorch streaming model (`tests/integration/test_export_parity.py`, agreement within 2e-4).
* Data: 2,000 four-second scenes rendered through the pipeline's own stage 1 + AGC (`StreamingPipeline.stage1`), so the model trains on exactly what it receives at run time; 200 validation scenes with different seeds. Speech from LibriSpeech dev-clean (the evaluation set uses test-clean). 20 % S1 / 40 % S2 / 40 % S3.
* Recipe: the upstream loss (compressed-magnitude MSE 0.3, compressed-complex MSE 0.2, STFT-consistency 0.3, waveform L1 0.2), AdamW, lr 2e-4 cosine to 1e-5, 8 epochs, +/-4 dB gain augmentation, about 10 minutes on an RTX 4050.
* Validation SI-SDR on the cascade input: **-1.68 dB (stock) -> 4.08 dB**; loss 0.0773 -> 0.0595, flattening over the last epochs.
* **Caveat:** the training and evaluation scenes come from the *same generator* (different seeds and speakers). The gain is real on this distribution; how much carries over to real recordings is unknown and must be checked on recorded data before relying on it.
* **Known trade-off:** at the quietest band the fine-tuned system has *lower* SI-SDR than AI-only (about 9 vs 14 dB) while ESTOI, PESQ, pitch and rhythm are better: it reshapes the waveform more when there is little noise to remove.

## 6. Compute and latency

Measured with `scripts/benchmark_pi5.py` on the **development PC** (single process, quiet machine, 30 s each). These say nothing about the Raspberry Pi 5, which must be benchmarked on the board (the script gates on the 99.9th percentile against the 10.67 ms deadline).

| preset | mean ms/block | p99.9 ms | RTF | stage 1 ms | neural ms |
|---|---|---|---|---|---|
| B0 (legacy) | 1.95 | 5.31 | 0.18 | 1.22 | 0.70 |
| bypass (AI only) | 0.68 | 1.48 | 0.06 | 0.00 | 0.64 |
| **ft_gsc_blend_agc12** | **1.13** | **2.90** | **0.11** | **0.30** | 0.71 |

The final pipeline's stage 1 is about 4x cheaper than the NLMS it replaces, and the whole block costs 58 % of B0's time. (Timings inside the 10-worker evaluation runs are inflated by contention and are not used for this comparison.)

**Algorithmic latency** is measured by the evaluator: 512 samples (10.7 ms) for the neural stage; the final pipeline measures 584 samples (12.2 ms) = 512 model + 24 beamformer alignment + 48 limiter lookahead. Add the capture and playback blocks (10.7 ms each at 512 samples) for a realistic lower bound of about 33 ms plus compute and OS buffering (see `docs/HARDWARE_ARCHITECTURE.md`). The earlier "< 22 ms glass-to-glass" claim is not supported.

## 7. Hardware sweep (what the simulation says about the pod)

Same 24 real-speech scenes (one per noise group x SPL band) re-rendered under each hardware variant, final pipeline vs AI-only. **Indicative, not definitive:** 24 scenes per variant (12 in the high-SPL bands); differences below about 0.03 ESTOI are within noise. The neural stage was fine-tuned on 120 dB-AOP, 1-2 cm, 4.5-6.5 cm data.

| variant | estoi_all | si_sdr_all | estoi_120_138 | si_sdr_120_138 | f0_recall_120_138 | ai_only_estoi_120_138 |
|---|---|---|---|---|---|---|
| base (AOP120, d1.5, r5.5) | 0.569 | 3.287 | 0.377 | -0.893 | 0.513 | 0.084 |
| AOP130 (same self-noise) | 0.599 | 5.712 | 0.428 | 2.213 | 0.522 | 0.140 |
| AOP130 (87 dB range) | 0.599 | 5.711 | 0.428 | 2.209 | 0.522 | 0.140 |
| AOP135 (same self-noise) | 0.611 | 6.307 | 0.467 | 3.500 | 0.548 | 0.147 |
| AOP135 (87 dB range) | 0.611 | 6.309 | 0.467 | 3.501 | 0.548 | 0.147 |
| spacing 0.8 cm | 0.555 | 2.029 | 0.369 | -1.992 | 0.480 | 0.084 |
| spacing 2.0 cm | 0.580 | 3.582 | 0.391 | -0.954 | 0.534 | 0.084 |
| spacing 3.0 cm | 0.579 | 3.309 | 0.381 | -1.674 | 0.549 | 0.084 |
| lips 4.5 cm | 0.601 | 3.480 | 0.417 | -0.418 | 0.557 | 0.094 |
| lips 6.5 cm | 0.542 | 2.717 | 0.339 | -1.753 | 0.477 | 0.076 |
| lips 4.0 cm, designed 5.5 | 0.614 | 3.621 | 0.434 | -0.351 | 0.592 | 0.097 |
| lips 7.0 cm, designed 5.5 | 0.532 | 2.355 | 0.326 | -2.100 | 0.456 | 0.072 |


**What it suggests**

* **Microphone overload point (AOP):** raising it from 120 to 130 / 135 dB lifts ESTOI at 120-138 dB SPL by about +0.05 / +0.09 and SI-SDR by about +3 / +4.4 dB, even though the model was not retrained for those captures. ("Same self-noise" and "87 dB range" are identical to three decimals: at these SPLs the mic's self-noise is irrelevant.) This is the most direct hardware lever against the 130 dB problem.
* **Distance to the lips:** closer is better, roughly +0.03 ESTOI per centimetre overall (4.5 cm vs 5.5 cm vs 6.5 cm), and the pipeline is insensitive to *designing* for the wrong distance (a wearer at 4 cm with a 5.5 cm design still does better than at 5.5 cm). Put the primary capsule as close to the mouth as is comfortable.
* **Capsule spacing:** 2-3 cm is no worse than 1.5 cm, and 0.8 cm is slightly worse (about -0.014 ESTOI, -1.3 dB SI-SDR); within the noise for this sample size. Avoid going below about 1 cm.
* **Where it still breaks:** even with a 135 dB overload point the 130-138 dB band stays well below the quiet bands (see the band table in `artifacts/eval/hardware_sweep/report.md`).

## 8. What is not verified

* Real recordings: everything is simulated. The noise models are literature-typical but invented; the fine-tuned model has only seen this generator.
* The Raspberry Pi 5: no timing, thermal or I2S/ALSA measurement was possible here.
* Live audio I/O: the capture, drift-compensation and UI code is unit-tested with simulated callbacks, not with real devices or a real I2S card.
* Calibration needs a quiet-room utterance; it improved the un-fine-tuned GSC by about 1.2 dB SI-SDR but was not combined with the fine-tuned model (which was trained without it).
* The vehicle-cabin (`pyroomacoustics`) environment from the plan was not generated.
* Channel-diversity declipping was not implemented: at the levels that matter both capsules clip on the same samples (clip fractions at the two mics differ by about one percentage point), so there is nothing to borrow.

## 9. Reproduce

```bash
uv run python scripts/download_data.py
uv run python scripts/generate_mixtures.py --set eval_v1 --workers 8
uv run python scripts/evaluate.py --set eval_v1 --workers 8 \
    --pipelines B0,bypass,gsc_blend,gsc_blend_agc12,ft_gsc_blend_agc12,adiff_blend,gsc_blend_cal --baseline bypass
uv run python scripts/make_train_set.py --preset gsc_blend_agc12 --n 2000 --n-val 200
uv run python scripts/train.py --data data/generated/train_gsc_blend_agc12 --epochs 8
uv run python scripts/export_fastenhancer_b.py --checkpoint artifacts/finetune/best.pth \
    --output models/fastenhancer/b/ft_gsc_blend_agc12.onnx
uv run python scripts/hardware_sweep.py --pipeline ft_gsc_blend_agc12
uv run python scripts/benchmark_pi5.py --preset ft_gsc_blend_agc12 --seconds 30   # run on the Pi
```
