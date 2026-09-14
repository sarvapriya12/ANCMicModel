<img src="docs/army-soldiers.jpg" alt="Dhwani-Kavach Tactical Microphone" width="258" />
# Dhwani-Kavach

**High-SPL Dual-Microphone Speech Enhancement System**

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)![Hardware Target](https://img.shields.io/badge/hardware-Raspberry_Pi_5_%7C_ESP32--S3-orange)![Status](https://img.shields.io/badge/status-active-success)![License](https://img.shields.io/badge/license-MIT-green)\---

## Overview

**Dhwani-Kavach** (meaning "Sound Shield" in Sanskrit) is an advanced, dual-microphone speech enhancement pipeline built for extreme acoustic environments. Designed for active battlefields, industrial zones, and aviation operations, the system can extract highly intelligible speech from up to **130 dB of background noise**.

By combining a deterministic Adaptive Filter (NLMS/FDAF) with a lightweight, causal Deep Learning model (FastEnhancer), Dhwani-Kavach operates entirely on edge hardware in real-time with sub-22ms glass-to-glass latency.

---

## ✨ Key Features

- **Dual-Mic Architecture:** Utilizes a Primary (Voice) and Reference (Noise) microphone array for spatial noise isolation.
- **Hybrid DSP + AI Pipeline:**
  - *Stage 1:* Adaptive Noise Subtraction (NLMS/FDAF) removes the bulk ambient field.
  - *Stage 2:* Neural Enhancement (FastEnhancer ONNX) eliminates non-linear transients and clipping artifacts.
- **Edge Optimized:** Achieves an RTF (Real-Time Factor) of \~0.512 on a Raspberry Pi 5.
- **Hardware Agnostic Acquisition:** Compatible with ESP32-S3 I2S MEMS arrays for highly synchronized, phase-coherent audio streaming over USB.

---

## Project Architecture

To keep this repository clean and maintainable, the source code is modularized. For deep dives into how the system works, refer to our core documentation:

-  **Hardware Architecture**: ESP32-S3 acquisition, DMA streaming, and microphone wiring diagrams.
- **Mathematical Framework**: Conceptual breakdown of the signal pathways, coherence gating, and custom neural loss functions.

```text
ANCMicModel/
├── docs/                   # System architecture and mathematical frameworks
├── esp32_firmware/         # C/C++ firmware for ESP32-S3 hardware acquisition
├── laboratory/             # Real-time testing UI and audio capture tools
├── scripts/                # Benchmark and deployment utilities
├── src/
│   └── highspl/            # Core Python package
│       ├── dsp/            # Adaptive filters (NLMS, FDAF)
│       ├── evaluation/     # Metrics (SI-SDR, ERLE, Coherence)
│       ├── models/         # Neural network inference wrappers
│       └── training/       # Custom loss functions for model fine-tuning
└── tests/                  # Unit and integration tests
```

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Raspberry Pi 5 (Deployment) or Windows/Linux PC (Development)
- PyTorch and ONNXRuntime

### 1. Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/your-org/ANCMicModel.git
cd ANCMicModel
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Running the Laboratory UI

To launch the real-time PyQt6 testing interface and visualize the noise cancellation:

```bash
python -m laboratory.ui.main_window
```

### 3. Running Benchmarks

To test the inference speed and latency on your specific hardware (e.g., Raspberry Pi 5):

```bash
python scripts/benchmark_pi5.py
```

---

## 🤝 Contributing

We welcome contributions to improve the DSP algorithms, optimize the ONNX models, or refine the ESP32 firmware. Please ensure that all new code is covered by `pytest` and adheres to the existing architectural patterns.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

*Developed for extreme environments. Silence the chaos.*
