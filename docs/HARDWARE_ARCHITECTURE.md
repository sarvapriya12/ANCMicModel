# Hardware architecture: two I2S MEMS microphones on a Raspberry Pi 5

```
 INMP441 #A  (primary, faces the lips)      L/R -> GND  (left  slot)
 INMP441 #B  (reference, faces away)        L/R -> 3V3  (right slot)
        \                                  /
         +---- BCLK / LRCLK / SD (shared) -+---->  Raspberry Pi 5 I2S (RP1)
                                                    |
                         ALSA stereo capture, 48 kHz, 24-bit data
                                                    |
                    highspl.StreamingPipeline (guard -> stage 1 -> AGC -> neural -> limiter)
                                                    |
                         headphones / radio  (separate audio device)
```

There is **no microcontroller, no firmware and no USB hop**. Both microphones are
the left/right slots of one I2S bus clocked by the Pi, so the two channels are
sample-synchronous by construction and arrive as one stereo ALSA stream.

> **Verification status.** Everything below about the *signal* is covered by the
> simulator and tests in this repository. The Pi-side ALSA / device-tree setup is
> standard practice but **has not been run on a real Pi 5 from this repository**:
> verify it on your OS image (see "Bring-up checklist").

## Wiring (3 signal wires + power)

| INMP441 pin | Mic A (primary) | Mic B (reference) | Pi 5 header |
|---|---|---|---|
| VDD | 3V3 | 3V3 | pin 1 (3.3 V) |
| GND | GND | GND | pin 6 |
| SCK (BCLK) | shared | shared | GPIO 18 (pin 12) |
| WS (LRCLK) | shared | shared | GPIO 19 (pin 35) |
| SD (data) | shared | shared | GPIO 20 (pin 38, PCM_DIN) |
| L/R | **GND** (left) | **3V3** (right) | - |

Both SD lines join one GPIO: each INMP441 tri-states its output outside its own
slot, selected by L/R. Keep the SD/BCLK/LRCLK stubs short and equal length;
decouple each mic with 100 nF at the pin.

## What the I2S choice does and does not guarantee

* **Sample clock identical for both mics** (shared BCLK/LRCLK), so there is no
  clock drift *between* the two channels. This is what an adaptive/differential
  stage needs.
* **Not literally "0 ms offset".** The L and R slots are one half-frame apart, so
  the reference trails by up to ~1 sample (~10-20 us). The offset is constant,
  and the pipeline's calibrator (`highspl.dsp.calibration`) measures and removes
  it; the simulator injects a random 0-1 sample offset in every scene.
* **Capture and playback are different clocks** unless playback is also on the
  Pi's I2S (e.g. an I2S DAC). USB / Bluetooth / HDMI audio drifts against the
  capture clock by tens of ppm, which would click every few minutes. The lab app
  handles this with `highspl.dsp.drift.DriftCompensator`.

## Acoustic design: a back-to-back, near-field endfire pair

```
   rear mic (B)      front mic (A)                       lips
   [ ]<--- d --->[ ]                                      o
   faces away    faces the lips         <------ r1 ------>
```

Measured/assumed geometry used throughout the simulator: spacing `d` 1-2 cm,
lips-to-primary `r1` 4.5-6.5 cm (the pod is 5-6 cm overall).

* Mouth speech reaches the rear mic **later and weaker**: delay `d/c` (~2 samples
  at 48 kHz) and amplitude `r1/(r1+d)` (~0.79, i.e. only **2 dB** less).
* A far-field noise wave from the front looks identical to speech except for that
  2 dB; noise from the **rear** arrives at the rear mic first and is cancelled by
  a cardioid (null behind); diffuse noise is coherent up to ~`c/(2d)` (8-11 kHz).
* Consequences (all measured on the synthetic bench, `docs/EVAL_REPORT.md`):
  a speech gate that waits for a 12 dB level difference can never fire, a classic
  NLMS "noise reference" cancels the talker, and **gain/timing match between the
  two capsules matters** (INMP441 sensitivity is +/-1 dB).

## Loudness: the 130 dB question

The INMP441's acoustic overload point is **120 dB SPL** (digital full scale).
Above it the capture clips; clipped speech cannot be recovered by any linear or
neural stage because the information is gone. The bench quantifies this with the
`aop_db` parameter (see the AOP sweep in `docs/EVAL_REPORT.md`) so the choice of
microphone for a 130 dB environment is made from data. Software can only detect
and contain clipping (adaptation freeze, AGC, limiter), not undo it.

### What the simulation says about the pod (indicative; `docs/EVAL_REPORT.md` section 7)

Final pipeline vs AI-only on the same 24 real-speech scenes per variant. The
differences below about 0.03 ESTOI are within noise at this sample size, and the
model was fine-tuned on 120 dB-AOP / 1-2 cm / 4.5-6.5 cm data.

| lever | change | effect at 120-138 dB SPL | overall |
|---|---|---|---|
| **overload point (AOP)** | 120 -> 130 / 135 dB | ESTOI 0.38 -> 0.43 / 0.47; SI-SDR -0.9 -> +2.2 / +3.5 dB | ESTOI 0.57 -> 0.60 / 0.61 |
| **distance to the lips** | 6.5 / 5.5 / 4.5 cm | ESTOI 0.34 / 0.38 / 0.42 | 0.54 / 0.57 / 0.60 |
| capsule spacing | 0.8 / 1.5 / 2 / 3 cm | 0.37 / 0.38 / 0.39 / 0.38 | 0.56 / 0.57 / 0.58 / 0.58 |

* **Move the primary capsule as close to the mouth as is comfortable** (about
  +0.03 ESTOI per centimetre). A wearer at 4 cm with a pipeline designed for 5.5 cm
  still does better than at 5.5 cm: the design distance barely matters.
* **A 130-135 dB-AOP microphone is the strongest hardware lever against the 130 dB
  problem** (+0.05 to +0.09 ESTOI where it matters), even without retraining. Even
  then the 130-138 dB band stays far below the quiet bands.
* Capsule spacing between 1.5 and 3 cm is not a sensitive choice; avoid going
  below about 1 cm.
* **Calibrate the pair.** A quiet-room utterance (`StreamingPipeline.calibrate`)
  lifted the un-fine-tuned GSC by about 1.2 dB SI-SDR (gain recovered to about
  3 %). It has not been combined with the fine-tuned model.

## Raspberry Pi bring-up checklist

1. Enable I2S in `/boot/firmware/config.txt` and add a capture-only I2S sound
   card overlay for generic I2S MEMS microphones (the common choices are the
   `googlevoicehat-soundcard` overlay or a vendor "i2smic" overlay; the exact
   name depends on the OS release - check what your image ships).
2. Confirm the card appears: `arecord -l`.
3. Record both channels: `arecord -D <card> -c 2 -r 48000 -f S32_LE -d 5 t.wav`
   (INMP441 data is 24-bit, MSB-aligned in a 32-bit slot). Each mic alone should
   show up on exactly one channel; tapping one mic must not affect the other.
4. Run the **quiet-room calibration** once (talk normally at the mouth position):
   `StreamingPipeline.calibrate(front, rear)` measures the mouth-path gain/delay
   and stores the correction for this device.
5. Check for xruns: the lab app's Health line shows input overflows / queue
   drops. Use small periods (128-256 frames) and a performance CPU governor.
6. Measure glass-to-glass latency with a loopback (play a click, record it): the
   lab app's figure is a *lower bound* (it excludes OS/driver buffering).

## Latency budget (what the code implies, not a promise)

| stage | samples | ms |
|---|---|---|
| capture block (ALSA period = block) | 512 | 10.7 |
| neural-stage algorithmic delay (**measured** by the evaluator) | 512 | 10.7 |
| limiter lookahead (if enabled) | 48 | 1.0 |
| front-end (differential/GSC alignment) | 8-24 | 0.2-0.5 |
| compute (per block, Pi-dependent) | - | benchmark it |
| playback block | 512 | 10.7 |
| **lower bound** | | **~33 ms + compute + OS buffering** |

The earlier "< 22 ms glass-to-glass" claim is not supported by this budget with
512-sample blocks. Smaller periods help the capture/playback terms, but the
neural stage's own 10.7 ms is fixed by its hop size.
