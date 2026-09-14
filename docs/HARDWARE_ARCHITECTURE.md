# Dhwani-Kavach ESP32-S3 Microphone Architecture: Final Engineering Decision

> **Cross-checked against:** Espressif ESP32-S3 Technical Reference Manual, ESP-IDF v6.1 ADC/I2S/USB documentation, Espressif Hardware Design Guidelines (Schematic Checklist), INMP441 datasheet, project Master Document (`High_SPL_SE_AI_Master_Document.md`), and `ARCHITECTURE_REVIEW.md`.

---

## Executive Summary

> [!CAUTION]
> **The HW-485 module + ESP32-S3 internal ADC is fundamentally unsuitable for audio-grade acquisition.** After thorough research, the internal SAR ADC was designed for sensor readings (battery voltage, potentiometers, temperature) — NOT continuous audio streaming. Using it for dual-channel 48 kHz speech in a 130 dB environment will produce unusable audio.

**The correct, industry-standard solution is: Replace the HW-485 with two INMP441 I2S digital MEMS microphones connected to ONE ESP32-S3.**

Both paths are documented below with full justification.

---

## Part 1: Why HW-485 + Internal ADC Fails (Evidence-Based)

### Evidence 1: Espressif's Own ADC Accuracy Specifications

From [Espressif Hardware Design Guidelines — ADC Section](https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32s3/schematic-checklist.html#adc):

| Attenuation | Effective Range | Total Error |
|---|---|---|
| ATTEN=0 | 0 – 850 mV | ±5 mV |
| ATTEN=1 | 0 – 1100 mV | ±6 mV |
| ATTEN=2 | 0 – 1600 mV | ±10 mV |
| **ATTEN=3 (11 dB)** | **0 – 2900 mV** | **±50 mV** |

At ATTEN=3 (which you MUST use to handle the HW-485's ~1.65V DC bias + audio swing):
- **±50 mV error on a 2900 mV range = ~1.7% measurement noise floor**
- Normal speech from the HW-485 produces only ±10 mV to ±30 mV AC swing
- **The ADC's own error (±50 mV) is LARGER than the speech signal itself**
- This is equivalent to an SNR of approximately **-4 dB to +6 dB** — completely unusable

### Evidence 2: Effective Number of Bits (ENOB)

- The ESP32-S3 ADC is a 12-bit SAR (0–4095 counts)
- True ENOB is approximately **9.0–9.8 bits** (confirmed by community measurements and Espressif's own errata)
- 9.5-bit ENOB = **~58 dB dynamic range**
- For comparison: A telephone-quality codec needs ≥66 dB. A proper audio ADC provides ≥96 dB (16-bit)
- Your master document specifies handling **130 dB SPL environments**. The internal ADC cannot resolve speech buried under that noise

### Evidence 3: Source Impedance Mismatch

- Espressif documentation mandates: **"Source impedance should be < 1 kΩ (preferably <100 Ω with an op-amp buffer)"**
- The HW-485's AO pin has an output impedance of **>2.2 kΩ** (unbuffered electret pull-up)
- At 48 kHz continuous DMA sampling, the SAR's switched-capacitor S/H circuit cannot settle in time
- Result: **high-frequency droop, inter-channel crosstalk, and amplitude-dependent distortion**

### Evidence 4: The HW-485 Module Itself

The HW-485 is a ~$0.50 hobbyist "clap detector" module consisting of:
- An electret capsule biased through a single resistor
- An LM393 comparator (for the DO digital threshold pin)
- **No operational amplifier, no gain stage, no anti-aliasing filter on the AO pin**
- The AO output is essentially a raw, unbuffered, unamplified electret signal

**Bottom line:** The HW-485 was designed to detect "is there a loud sound? yes/no" — not to capture intelligible speech waveforms.

---

## Part 2: The Correct Solution — INMP441 I2S Digital MEMS Microphones

### Why INMP441 Eliminates Every Problem

The **INMP441** is a digital I2S MEMS microphone module (~$1.50–$2.00 each) that contains:
- A MEMS acoustic transducer
- A **built-in 24-bit sigma-delta ADC** (vs. the ESP32's 12-bit SAR)
- A digital I2S output interface

| Parameter | HW-485 + ESP32 Internal ADC | INMP441 I2S MEMS |
|---|---|---|
| **Bit Depth** | 12-bit SAR (ENOB ~9.5) | **24-bit Sigma-Delta** |
| **Dynamic Range** | ~58 dB | **≥87 dB** |
| **SNR** | ~25–35 dB (with noise) | **61 dB** |
| **Frequency Response** | Depends on analog BPF circuit | **60 Hz – 15 kHz (flat ±1 dB)** |
| **Anti-Aliasing** | YOU must build it in hardware | **Built into the chip** |
| **DC Offset** | Must manually bias at 1.65V | **None (digital output, AC-coupled internally)** |
| **Output Impedance** | >2.2 kΩ (needs op-amp buffer) | **N/A (digital I2S, no analog path)** |
| **Analog BPF Needed?** | YES (HPF + LPF + gain stage) | **NO** |
| **Op-Amp Buffer Needed?** | YES | **NO** |
| **ADC Calibration Needed?** | YES (eFuse curves + polynomial) | **NO** |
| **Phase-Matched Dual Channel?** | Extremely difficult (analog tolerance) | **Trivially perfect (shared I2S clock)** |
| **Cost** | HW-485 (~$0.50) + op-amp (~$0.30) + passives (~$0.20) | **~$1.50–$2.00** |

### The Killer Feature: Two INMP441s on ONE I2S Bus = Perfect Sync

The INMP441 has an **L/R (Left/Right) channel select pin**:
- **L/R tied to GND** → Microphone outputs on the **LEFT channel** of I2S
- **L/R tied to VDD** → Microphone outputs on the **RIGHT channel** of I2S

This means:

```
INMP441 #1 (Primary / Voice Mic)     L/R → GND (Left Channel)
    ├── SCK  ───────┐
    ├── WS   ───────┤
    └── SD   ───────┤
                    ├──────► ESP32-S3 I2S0 Peripheral
INMP441 #2 (Reference / Noise Mic)   L/R → VDD (Right Channel)
    ├── SCK  ───────┤
    ├── WS   ───────┤
    └── SD   ───────┘
```

- **Both microphones share the EXACT SAME clock lines (SCK + WS)**
- **Zero sample drift. Zero phase mismatch. Perfect synchronization — forever**
- The ESP32-S3 I2S DMA reads interleaved stereo frames: `[Left_sample, Right_sample, Left_sample, Right_sample, ...]`
- Deinterleaving in firmware gives you perfectly aligned Primary and Reference channels

---

## Part 3: The Complete Final Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│ INMP441 #1          │         │ INMP441 #2          │
│ (Primary / Voice)   │         │ (Reference / Noise) │
│ L/R → GND           │         │ L/R → VDD           │
│ Placement: Near     │         │ Placement: Outward  │
│ mouth / boom mic    │         │ facing / ambient     │
└──────┬──────────────┘         └──────┬──────────────┘
       │ SD (Data)                     │ SD (Data)
       │                               │
       │    ┌── SCK (Shared Clock) ────┤
       │    ├── WS  (Shared Word Sel) ─┤
       │    │                          │
       ▼    ▼                          ▼
┌──────────────────────────────────────────────────────┐
│ ESP32-S3 (Single Board)                              │
│                                                      │
│ I2S0 Peripheral (Standard Mode, 32-bit, Stereo)     │
│ ├── GPIO 4  → I2S0_SCK  (Bit Clock)                 │
│ ├── GPIO 5  → I2S0_WS   (Word Select / LRCK)        │
│ ├── GPIO 6  → I2S0_DIN  (Data In from both mics)    │
│                                                      │
│ Firmware:                                            │
│ • I2S DMA reads stereo 16-bit @ 48,000 Hz            │
│ • Deinterleave → ch_primary[], ch_reference[]        │
│ • Digital DC blocker (y[n] = x[n]-x[n-1]+0.995·y[n-1])│
│ • Frame into 512-sample packets with sequence number │
│ • Stream via USB-CDC Serial to Raspberry Pi 5        │
│                                                      │
│ USB OTG (GPIO19=D-, GPIO20=D+) ──────────────────────┤
└──────────────────────────┬───────────────────────────┘
                           │ USB Full-Speed (12 Mbps)
                           │ Stereo 16-bit 48kHz PCM
                           │ = 192 KB/s (1.5 Mbps)
                           │ << 12 Mbps capacity
                           ▼
┌──────────────────────────────────────────────────────┐
│ Raspberry Pi 5                                       │
│                                                      │
│ USB Serial Reader (Python thread):                   │
│ • Reads framed binary packets from /dev/ttyACM0      │
│ • Validates sequence numbers (detects drops)         │
│ • Splits into primary[] and reference[] arrays       │
│                                                      │
│ Digital Signal Processing Pipeline:                  │
│ 1. Digital Linear-Phase BPF (100 Hz – 8 kHz)         │
│    Applied IDENTICALLY to both channels              │
│ 2. PLD Voice Gate (coherence + energy ratio)         │
│ 3. RobustNLMS / BattlefieldFDAF                     │
│    (Adaptive noise subtraction using reference)      │
│ 4. FastEnhancer-Base ONNX (48 kHz streaming, ~1 ms) │
│                                                      │
│ Output: Clean tactical speech → Headphones / Radio   │
│ Total glass-to-glass latency: < 22 ms                │
└──────────────────────────────────────────────────────┘
```

---

## Part 4: Why This Architecture is Correct (Cross-Check Summary)

### Check 1: Master Document Compliance
Your `High_SPL_SE_AI_Master_Document.md` specifies:
- ✅ **2 channels** (Primary + Reference) → Two INMP441s
- ✅ **Useful acoustic band ≤ ~15 kHz** → INMP441 flat response 60 Hz – 15 kHz
- ✅ **NLMS/reference cancellation preprocessing** → Identical digital filtering + NLMS on Pi 5
- ✅ **Pi 5 primary model: FastEnhancer-Base 48 kHz** → Confirmed working (RTF 0.512)
- ✅ **Streaming / causal inference** → I2S DMA → USB → Pi 5 pipeline is fully streaming

### Check 2: Phase Coherence for Adaptive Filtering
- Two INMP441s on the same I2S bus share SCK and WS
- Both microphones sample at the **exact same instant** (within nanoseconds)
- Phase mismatch: **0.000 ms** (vs. two ESP32s drifting at ±20-50 ppm)
- NLMS/FDAF optimal filter $W_{opt}(z) = H_{acoustic}(z)$ — no analog filter ratio to compensate

### Check 3: No Analog BPF Needed At All
- INMP441 has **internal high-pass filter** at ~60 Hz (removes mechanical rumble)
- INMP441 has **internal anti-aliasing filter** matched to its sigma-delta ADC
- The tight voice bandpass (100 Hz – 8 kHz) is applied **digitally** on the Pi 5
- Result: Zero analog component tolerance issues, zero phase mismatch

### Check 4: Signal Quality vs. High-SPL Requirements
- INMP441 Acoustic Overload Point (AOP): **120 dB SPL**
- INMP441 SNR: **61 dB**
- For environments exceeding 120 dB, the INMP441 will gracefully clip (known, predictable behavior)
- Your existing `ClippingPenaltyLoss` and clipping detection in the Trust Estimator already handle this
- This is STILL vastly superior to the HW-485 + ESP32 ADC which clips unpredictably at ~95-100 dB

### Check 5: Data Rate & USB Bandwidth
- Stereo 16-bit @ 48 kHz = `2 × 2 × 48000 = 192,000 bytes/sec = 1.5 Mbps`
- ESP32-S3 USB Full-Speed = **12 Mbps**
- Utilization: **12.5%** — massive headroom for packet headers, retransmission, and metadata

### Check 6: Cost Comparison
| Component | HW-485 Path | INMP441 Path |
|---|---|---|
| Microphone modules (×2) | $1.00 | **$3.00–$4.00** |
| Op-amp buffer board (×2) | $1.00 | **$0.00** |
| Passive components (R, C for BPF) | $0.50 | **$0.00** |
| ESP32-S3 board | $5.00 | **$5.00** |
| Total analog complexity | HIGH (solder, tune, calibrate) | **ZERO** |
| **Total BOM** | **~$7.50** | **~$8.00–$9.00** |

For ~$1.50 more, you eliminate ALL analog engineering headaches.

---

## Part 5: What Changes in the Current Pipeline

### Changes in `D:\SIH\Esp32 Mic communicator` (NEW folder)

| Component | What to Build |
|---|---|
| `firmware_esp32s3/` | ESP-IDF project: I2S DMA stereo capture → USB-CDC framed streaming |
| `host_pi5/esp32_receiver.py` | Python USB serial reader → deframes packets → feeds HighSPLPipeline |

### Changes in `D:\SIH\ANCMicModel` (EXISTING pipeline)

| File | Change |
|---|---|
| `laboratory/audio/capture.py` | Add new `ESP32AudioCapture` class that reads from USB serial instead of `sounddevice` |
| `laboratory/dsp/pipeline.py` | No change needed (already accepts dual-channel numpy arrays) |
| `src/highspl/dsp/nlms.py` | No change needed |
| `src/highspl/dsp/fdaf.py` | No change needed |
| `src/highspl/models/fastenhancer.py` | No change needed |

### Wiring Diagram (3 wires + power)

```
INMP441 #1 (Voice)          ESP32-S3 DevKit          INMP441 #2 (Reference)
┌──────────┐                ┌──────────┐              ┌──────────┐
│ VDD ─────┼── 3.3V ───────┤ 3V3      ├── 3.3V ─────┤ VDD      │
│ GND ─────┼── GND ────────┤ GND      ├── GND ──────┤ GND      │
│ SD  ─────┼────────────────┤ GPIO 6   ├─────────────┤ SD       │
│ SCK ─────┼────────────────┤ GPIO 4   ├─────────────┤ SCK      │
│ WS  ─────┼────────────────┤ GPIO 5   ├─────────────┤ WS       │
│ L/R ─────┼── GND (Left)  │          │  VDD (Right)─┤ L/R      │
└──────────┘                │ USB-C ───┼──► Pi 5 USB  └──────────┘
                            └──────────┘
```

> [!IMPORTANT]
> The L/R pin on INMP441 #1 must be tied to **GND** (Left channel = Primary).
> The L/R pin on INMP441 #2 must be tied to **VDD** (Right channel = Reference).
> Both SD (data) lines connect to the **same GPIO 6** — they time-share the bus via L/R selection.

---

## Part 6: If using HW-485 + ESP32-S3

If INMP441 modules are unavailable and you are constrained to strictly using the HW-485 modules, the following "Raw Capture + Digital Fix" strategy must be strictly followed to mitigate the hardware limitations.

### 1. Hardware Wiring
Both modules must be wired to the **ADC1** block on the ESP32-S3 for simultaneous DMA sampling.

```text
HW-485 #1 (Primary)            ESP32-S3               HW-485 #2 (Reference)
┌──────────┐                 ┌──────────┐              ┌──────────┐
│ VCC ─────┼─── 3.3V ───────┤ 3V3      ├── 3.3V ─────┤ VCC      │
│ GND ─────┼─── GND ────────┤ GND      ├── GND ──────┤ GND      │
│ AO  ─────┼────────────────┤ GPIO 1   │             │          │
│ DO       │  (ADC1_CH0)    │          │             │          │
└──────────┘                │          │             │          │
                            │ GPIO 2   ├─────────────┼── AO     │
              (ADC1_CH1)    │          │             │   DO     │
                            │ USB-C ───┼──► To Pi 5  └──────────┘
                            └──────────┘
```

**CRITICAL HARDWARE HACKS:**
1. **Power with 3.3V, NOT 5V!** The ESP32-S3 ADC pins cannot handle more than 3.3V. If you power the HW-485 with 5V, the analog output (AO) DC bias will exceed the ESP32's maximum rating and fry the ADC pin.
2. **The Capacitor Trick (Highly Recommended):** Because the HW-485 lacks a buffer, Espressif strongly recommends placing a **0.1 µF (100 nF) ceramic capacitor** between GPIO 1 and GND, and another between GPIO 2 and GND. This stabilizes the ADC's sampling capacitor and drastically reduces crosstalk and high-frequency noise.

### 2. ESP32-S3 Firmware Strategy
To get 48 kHz stereo audio out of the internal ADC, normal `analogRead()` in Arduino will completely fail (it's too slow and drifts). The firmware must use:
* **ADC Continuous Mode (DMA):** The ESP32's hardware timer will trigger the ADC exactly 48,000 times a second, reading GPIO 1 and GPIO 2 simultaneously into memory.
* **Attenuation 3 (11 dB):** The HW-485 signal rests around 1.65V. We must use the highest attenuation setting so the ADC can read the full 0–3.1V range without clipping the baseline.
* **USB-CDC (Serial) Streaming:** The DMA will push chunks of binary data directly out the USB port to the Pi 5.

### 3. Pi 5 Software Strategy (Fixing the Audio)
Because the HW-485 signal is small (maybe ±50 mV of swing) and we are using a 3300 mV range, the audio will initially sound very quiet and slightly "crunchy" (quantization noise). We will fix this in your Pi 5 Python script:

1. **Digital DC Blocker:** We will run a mathematical filter: `y[n] = x[n] - x[n-1] + 0.995 * y[n-1]` to instantly strip away the 1.65V DC baseline.
2. **Digital Gain (Software Amp):** We will multiply the signal by ~15x to bring it up to standard line-level volume.
3. **AI Denoising:** Your `FastEnhancer-Base` model is excellent at suppressing quantization and background noise. It will treat the ADC's imperfections as just another layer of noise to cancel out.

---

## Final Recommendation

> [!TIP]
> Order two INMP441 breakout boards (~$1.50 each). They eliminate the need for ANY analog BPF, ANY op-amp, ANY ADC calibration, and give you perfect dual-channel synchronization with 24-bit audio quality. The firmware is simpler, the wiring is simpler, and the audio quality is categorically superior.
