from dataclasses import dataclass

import numpy as np


@dataclass
class NLMSState:
    """State carried between streaming NLMS process calls."""
    weights: np.ndarray
    x_hist: np.ndarray
    energy_ema: float
    primary_energy_ema: float
    warmup_samples: int
    freeze_remaining: int = 0

class RobustNLMS:
    """Highly Robust NLMS Adaptive Filter with advanced safety gating."""
    def __init__(self, filter_length: int = 256, step_size: float = 0.25, leakage: float = 1e-5, freeze_ratio_db: float = 12.0, pld_threshold_db: float = 12.0, freeze_on_clipping: bool = True, freeze_on_divergence: bool = True, min_reference_energy: float = -40.0, max_reference_energy: float = -3.0, eps: float = 1e-8) -> None:
        self.n = filter_length
        self.mu = step_size
        self.leakage = leakage
        self.freeze_ratio = 10.0 ** (freeze_ratio_db / 10.0)
        self.pld_threshold = 10.0 ** (pld_threshold_db / 10.0)
        self.freeze_on_clipping = freeze_on_clipping
        self.freeze_on_divergence = freeze_on_divergence
        self.min_ref_power = 10.0 ** (min_reference_energy / 10.0)
        self.max_ref_power = 10.0 ** (max_reference_energy / 10.0)
        self.eps = eps

    def reset(self) -> NLMSState:
        return NLMSState(weights=np.zeros(self.n, dtype=np.float32), x_hist=np.zeros(self.n, dtype=np.float32), energy_ema=self.eps, primary_energy_ema=self.eps, warmup_samples=0, freeze_remaining=0)

    def process(self, primary: np.ndarray, reference: np.ndarray, state: NLMSState) -> tuple[np.ndarray, NLMSState]:
        d = np.asarray(primary, dtype=np.float32)
        x = np.asarray(reference, dtype=np.float32)

        if d.ndim != 1 or x.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays")
        if len(d) != len(x):
            raise ValueError("primary and reference length mismatch")
        if len(d) == 0:
            return np.empty(0, dtype=np.float32), state

        out = np.empty_like(d)
        w = state.weights
        energy = float(state.energy_ema)
        p_energy = float(state.primary_energy_ema)
        warmup = int(state.warmup_samples)

        block_energy = energy
        block_frozen = False
        for i, sample in enumerate(x):
            inst_energy = float(sample) ** 2
            block_energy = 0.995 * block_energy + 0.005 * inst_energy
            bias_correction = 1.0 - 0.995 ** (warmup + i + 1)
            spike_ratio = inst_energy / ((block_energy / bias_correction) + self.eps)
            if spike_ratio > self.freeze_ratio:
                block_frozen = True
                break

        extended_x = np.concatenate((state.x_hist[::-1], x))
        hist_norm = float(np.dot(state.x_hist, state.x_hist)) + self.eps
        freeze_remaining = int(state.freeze_remaining)

        for i in range(len(x)):
            hist = extended_x[i + 1 : i + 1 + self.n][::-1]
            ref_val = float(x[i])
            prim_val = float(d[i])
            inst_energy = ref_val ** 2
            p_inst_energy = prim_val ** 2
            energy = 0.995 * energy + 0.005 * inst_energy
            p_energy = 0.995 * p_energy + 0.005 * p_inst_energy

            y_hat = float(np.dot(w, hist))
            error = prim_val - y_hat
            out[i] = error

            oldest_val = float(extended_x[i])
            hist_norm = hist_norm + inst_energy - (oldest_val ** 2)
            hist_norm = max(hist_norm, self.eps)
            if freeze_remaining > 0: freeze_remaining -= 1; continue
            if block_frozen: continue

            skip_adapt = False
            if (p_energy / (energy + self.eps)) > self.pld_threshold: skip_adapt = True
            if self.freeze_on_clipping and (abs(ref_val) > 0.99 or abs(prim_val) > 0.99): skip_adapt = True
            if energy < self.min_ref_power or energy > self.max_ref_power: skip_adapt = True
            if self.freeze_on_divergence and abs(error) > (abs(prim_val) * 3.0 + 0.01): skip_adapt = True

            if skip_adapt: continue
            if self.leakage > 0.0: w *= 1.0 - self.mu * self.leakage
            w += (self.mu * error / hist_norm) * hist

        if block_frozen: freeze_remaining = self.n
        state.x_hist = extended_x[-self.n :][::-1]
        state.energy_ema = energy
        state.primary_energy_ema = p_energy
        state.warmup_samples += len(x)
        state.freeze_remaining = freeze_remaining
        return out, state


class NLMS:
    """Standard canonical baseline Normalized Least Mean Squares (NLMS) adaptive filter."""
    def __init__(
        self,
        filter_length: int = 256,
        step_size: float = 0.25,
        leakage: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        self.n = filter_length
        self.mu = step_size
        self.leakage = leakage
        self.eps = eps

    def reset(self) -> NLMSState:
        return NLMSState(
            weights=np.zeros(self.n, dtype=np.float32),
            x_hist=np.zeros(self.n, dtype=np.float32),
            energy_ema=self.eps,
            primary_energy_ema=self.eps,
            warmup_samples=0,
            freeze_remaining=0,
        )

    def process(self, primary: np.ndarray, reference: np.ndarray, state: NLMSState) -> tuple[np.ndarray, NLMSState]:
        d = np.asarray(primary, dtype=np.float32)
        x = np.asarray(reference, dtype=np.float32)

        if d.ndim != 1 or x.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays")
        if len(d) != len(x):
            raise ValueError("primary and reference length mismatch")
        if len(d) == 0:
            return np.empty(0, dtype=np.float32), state

        out = np.empty_like(d)
        w = state.weights
        energy = float(state.energy_ema)
        p_energy = float(state.primary_energy_ema)

        extended_x = np.concatenate((state.x_hist[::-1], x))
        hist_norm = float(np.dot(state.x_hist, state.x_hist)) + self.eps

        for i in range(len(x)):
            hist = extended_x[i + 1 : i + 1 + self.n][::-1]
            ref_val = float(x[i])
            prim_val = float(d[i])
            inst_energy = ref_val ** 2
            p_inst_energy = prim_val ** 2
            energy = 0.995 * energy + 0.005 * inst_energy
            p_energy = 0.995 * p_energy + 0.005 * p_inst_energy

            y_hat = float(np.dot(w, hist))
            error = prim_val - y_hat
            out[i] = error

            oldest_val = float(extended_x[i])
            hist_norm = hist_norm + inst_energy - (oldest_val ** 2)
            hist_norm = max(hist_norm, self.eps)

            if self.leakage > 0.0:
                w *= 1.0 - self.mu * self.leakage
            w += (self.mu * error / hist_norm) * hist

        state.x_hist = extended_x[-self.n :][::-1]
        state.energy_ema = energy
        state.primary_energy_ema = p_energy
        state.warmup_samples += len(x)
        return out, state


@dataclass
class VSSNLMSState:
    """State for Variable Step-Size NLMS"""
    weights: np.ndarray
    x_hist: np.ndarray
    mu_current: float
    hist_norm: float

class VSSNLMS:
    """Academic Variable Step-Size NLMS (VSS-NLMS).
    Dynamically adjusts learning rate based on error power, offering a smooth 
    tradeoff between convergence speed and steady-state error.
    """
    def __init__(self, filter_length: int = 256, mu_max: float = 0.5, mu_min: float = 0.001, alpha: float = 0.99, gamma: float = 0.01, eps: float = 1e-8):
        self.n = filter_length
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.alpha = alpha  # Memory factor (how fast mu decays)
        self.gamma = gamma  # Sensitivity to error power
        self.eps = eps

    def reset(self) -> VSSNLMSState:
        return VSSNLMSState(
            weights=np.zeros(self.n, dtype=np.float32),
            x_hist=np.zeros(self.n, dtype=np.float32),
            mu_current=self.mu_max,
            hist_norm=self.eps
        )

    def process(self, primary: np.ndarray, reference: np.ndarray, state: VSSNLMSState) -> tuple[np.ndarray, VSSNLMSState]:
        d = np.asarray(primary, dtype=np.float32)
        x = np.asarray(reference, dtype=np.float32)

        if d.ndim != 1 or x.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays")
        if len(d) != len(x):
            raise ValueError("primary and reference length mismatch")
        if len(d) == 0:
            return np.empty(0, dtype=np.float32), state

        out = np.empty_like(d)
        
        w = state.weights
        mu = state.mu_current
        extended_x = np.concatenate((state.x_hist[::-1], x))
        hist_norm = state.hist_norm

        for i in range(len(x)):
            hist = extended_x[i + 1 : i + 1 + self.n][::-1]
            ref_val = float(x[i])
            prim_val = float(d[i])
            inst_energy = ref_val ** 2

            # Predict and calculate error
            y_hat = float(np.dot(w, hist))
            error = prim_val - y_hat
            out[i] = error

            # Update history norm
            oldest_val = float(extended_x[i])
            hist_norm = hist_norm + inst_energy - (oldest_val ** 2)
            hist_norm = max(hist_norm, self.eps)

            # Update Variable Step-Size (mu)
            error_power = error ** 2
            mu = self.alpha * mu + self.gamma * error_power
            mu = min(mu, self.mu_max)
            mu = max(mu, self.mu_min)

            # Update weights
            w += (mu * error / hist_norm) * hist

        state.x_hist = extended_x[-self.n :][::-1]
        state.mu_current = mu
        state.hist_norm = hist_norm
        return out, state
