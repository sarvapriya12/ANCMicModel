# Mathematical Framework: High-SPL Speech Enhancement

This document outlines the conceptual mathematical framework used in the Dhwani-Kavach High-SPL Speech Enhancement system. It details the variables and signal pathways used for our hybrid Adaptive Filter (NLMS/FDAF) + Deep Learning approach, without disclosing the proprietary algorithmic implementations.

## 1. Core Signal Definitions

The system utilizes a dual-microphone setup to isolate speech from extreme ambient noise (e.g., 130 dB battlefield environments).

*   **$d(n)$**: The **Primary Signal** (Voice Target). This is the signal captured by the microphone closest to the user's mouth. It contains the desired speech $s(n)$ mixed with a large amount of ambient noise $n_p(n)$ and microphone characteristics $\epsilon_p(n)$.
    *   $d(n) = s(n) + n_p(n) + \epsilon_p(n)$
*   **$r(n)$**: The **Reference Signal** (Noise Target). This is captured by the outward-facing microphone. It ideally contains only the ambient noise field $n_r(n)$ and microphone characteristics $\epsilon_r(n)$, with minimal speech leakage.
    *   $r(n) = n_r(n) + \epsilon_r(n)$
*   **$s(n)$**: The clean, desired speech signal.
*   **$n(n)$**: The ambient noise field.
*   **$e(n)$**: The **Error Signal** (Residual). This is the output of the first stage of adaptive filtering, representing the primary signal after the estimated reference noise has been subtracted.

## 2. Stage 1: Adaptive Noise Subtraction

The first stage attempts to linearly subtract the noise from the primary signal using the reference signal.

### The Adaptive Filter
We employ an advanced adaptive filter algorithm—specifically designed as a **Delayless Variable Step-Size (VSS) Hybrid**—that fuses time-domain NLMS with frequency-domain Kalman filter-grade double-talk immunity. This models the acoustic path between the reference microphone and the primary microphone with **< 0.1 ms latency** and an efficient compute footprint (~32M MACS).

*   **$W(n)$**: The adaptive filter weight vector at time step $n$.
*   **$\hat{n}_p(n)$**: The estimated noise at the primary microphone, calculated by applying the filter weights to the reference signal.
    *   $\hat{n}_p(n) = W^T(n) r(n)$

### Error Calculation
The estimated noise is subtracted from the primary signal to yield the error signal. In an optimal scenario, $e(n) \approx s(n)$.
*   $e(n) = d(n) - \hat{n}_p(n)$

### Filter Update Mechanism
The filter weights $W(n)$ are continuously updated to minimize the power of the error signal $e(n)$ when speech is absent.
*   $W(n+1) = W(n) + \mu \cdot f(e(n), r(n))$
    *   **$\mu$ (Step Size)**: Controls how quickly the filter adapts to changes in the noise environment.
    *   **$f(\cdot)$**: The update function (specific implementations are proprietary).

### Control Logic (Voice Activity & Coherence)
To prevent the adaptive filter from canceling the desired speech (a phenomenon known as "signal cancellation"), the update mechanism is governed by:
*   **$\gamma(n)$ (Coherence)**: Measures the linear dependence between the primary and reference signals.
*   **$\Phi(n)$ (Energy Ratio)**: Compares the energy of the primary and reference channels to detect near-end speech.

## 3. Stage 2: Deep Learning Enhancement

The residual signal $e(n)$ still contains non-linear noise, transient artifacts, and residual acoustic leakage that the linear adaptive filter cannot resolve.

*   **$y(n)$**: The final enhanced output signal.
*   **$F_{\theta}(\cdot)$**: The deep neural network model (**FastEnhancer**, a causal RNNFormer architecture), parameterized by weights $\theta$.

The neural network takes the residual signal $e(n)$ and extracts the clean speech components:
*   $y(n) = F_{\theta}(e(n))$

### Loss Function Pipeline
During training, the neural network weights $\theta$ are optimized using a composite loss function $\mathcal{L}_{total}$:

*   **$\mathcal{L}_{STFT}$**: Multi-Resolution STFT Loss. Penalizes differences in the spectral domain across multiple time-frequency resolutions.
*   **$\mathcal{L}_{SISNR}$**: Scale-Invariant Signal-to-Noise Ratio Loss. Maximizes the target signal energy relative to the residual noise.
*   **$\mathcal{L}_{ERLE}$**: Echo Return Loss Enhancement penalty. Explicitly penalizes residual energy during periods when only noise (no speech) is present.
*   **$\mathcal{L}_{clip}$**: Clipping Penalty. Punishes the model for generating waveforms that exceed the dynamic range $[-1, 1]$.

$$ \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{STFT} + \lambda_2 \mathcal{L}_{SISNR} + \lambda_3 \mathcal{L}_{ERLE} + \lambda_4 \mathcal{L}_{clip} $$

This custom composite formulation uniquely enforces stable model convergence in extreme **130 dB SPL** battlefield acoustic environments, balancing perceptual speech fidelity against catastrophic dynamic range clipping.
