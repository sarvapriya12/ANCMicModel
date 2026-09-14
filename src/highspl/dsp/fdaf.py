from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class FDAFState:
    """State carried between streaming Partitioned Block FDAF process calls."""
    weights: np.ndarray             # shape: (num_partitions, fft_size), complex64
    reference_history: np.ndarray   # shape: (num_partitions, fft_size), complex64
    prev_ref: np.ndarray            # shape: (block_size,), float32
    p_x: np.ndarray                 # shape: (fft_size,), float32
    p_d: np.ndarray                 # shape: (fft_size,), float32
    p_xd: np.ndarray                # shape: (fft_size,), complex64
    buf_primary: np.ndarray         # leftover samples, float32
    buf_reference: np.ndarray       # leftover samples, float32
    primary_power: float = 0.0
    reference_power: float = 0.0
    warmup_samples: int = 0


class BattlefieldFDAF:
    """Partitioned Block Frequency Domain Adaptive Filter (Battlefield FDAF).

    Ported from BattlefieldAEC.hpp and DualMicBattlefieldAEC.hpp.
    Implements:
      - Overlap-save block FFT partitioned filtering
      - Partition weight history
      - Magnitude squared coherence tracking for adaptive step size
      - Hard clipping suspension and voice-dominant primary channel protection
      - Time-domain constraint projection (zeroing second half of impulse response)
    """

    def __init__(
        self,
        block_size: int = 128,
        num_partitions: int = 4,
        filter_length: Optional[int] = None,
        step_size: float = 0.05,
        alpha: float = 0.95,
        leakage: float = 0.0,
        eps: float = 1e-6,
        clip_threshold: float = 0.99,
        voice_protection: bool = True,
        voice_ratio: float = 1.5,
    ) -> None:
        self.block_size = int(block_size)
        self.fft_size = 2 * self.block_size

        if filter_length is not None:
            self.num_partitions = max(1, filter_length // self.block_size)
        else:
            self.num_partitions = int(num_partitions)

        self.step_size = float(step_size)
        self.alpha = float(alpha)
        self.leakage = float(leakage)
        self.eps = float(eps)
        self.clip_threshold = float(clip_threshold)
        self.voice_protection = bool(voice_protection)
        self.voice_ratio = float(voice_ratio)

    def reset(self) -> FDAFState:
        """Create a fresh initial FDAF state."""
        return FDAFState(
            weights=np.zeros((self.num_partitions, self.fft_size), dtype=np.complex64),
            reference_history=np.zeros((self.num_partitions, self.fft_size), dtype=np.complex64),
            prev_ref=np.zeros(self.block_size, dtype=np.float32),
            p_x=np.full(self.fft_size, self.eps, dtype=np.float32),
            p_d=np.full(self.fft_size, self.eps, dtype=np.float32),
            p_xd=np.zeros(self.fft_size, dtype=np.complex64),
            buf_primary=np.empty(0, dtype=np.float32),
            buf_reference=np.empty(0, dtype=np.float32),
            primary_power=0.0,
            reference_power=0.0,
            warmup_samples=0,
        )

    def _process_one_block(
        self,
        d_block: np.ndarray,
        x_block: np.ndarray,
        state: FDAFState,
    ) -> Tuple[np.ndarray, FDAFState]:
        """Process exactly one block of length block_size."""
        B = self.block_size
        N = self.fft_size

        # Overlap-save reference vector: [prev_ref, x_block]
        ref_buf = np.concatenate((state.prev_ref, x_block))
        state.prev_ref = x_block.copy()

        # FFT of reference buffer
        X_k = np.fft.fft(ref_buf).astype(np.complex64)

        # Shift partition reference history
        state.reference_history[1:] = state.reference_history[:-1]
        state.reference_history[0] = X_k

        # Pad desired mic block with B zeros in the first half
        padded_d = np.zeros(N, dtype=np.float32)
        padded_d[B:] = d_block
        D_k = np.fft.fft(padded_d).astype(np.complex64)

        # Filter estimation across all partitions
        Y_k = np.sum(state.weights * state.reference_history, axis=0)
        y_time = np.real(np.fft.ifft(Y_k)).astype(np.float32)

        # Clean output estimate for current block
        y_est = y_time[B:]
        e_block = d_block - y_est

        # Pad error with B zeros in first half for gradient
        padded_e = np.zeros(N, dtype=np.float32)
        padded_e[B:] = e_block
        E_k = np.fft.fft(padded_e).astype(np.complex64)

        # Power tracking
        alpha = self.alpha
        state.p_xd = alpha * state.p_xd + (1.0 - alpha) * (X_k * np.conj(D_k))
        state.p_d = alpha * state.p_d + (1.0 - alpha) * (np.abs(D_k) ** 2)
        state.p_x = alpha * state.p_x + (1.0 - alpha) * (np.abs(X_k) ** 2)

        # Coherence tracking
        coherence = (np.abs(state.p_xd) ** 2) / (state.p_x * state.p_d + self.eps)
        coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)

        # Clipping protection
        max_val = max(float(np.max(np.abs(d_block))), float(np.max(np.abs(x_block))))
        thresh = self.clip_threshold if max_val <= 1.5 else 32000.0
        is_clipped = max_val > thresh

        # Voice dominant protection
        p_pow = float(np.mean(d_block ** 2))
        r_pow = float(np.mean(x_block ** 2))
        state.primary_power = 0.99 * state.primary_power + 0.01 * p_pow
        state.reference_power = 0.99 * state.reference_power + 0.01 * r_pow
        voice_dominant = (
            self.voice_protection and
            (state.primary_power > state.reference_power * self.voice_ratio)
        )

        if is_clipped or voice_dominant:
            mu_k = np.zeros(N, dtype=np.float32)
        else:
            mu_k = (self.step_size * coherence).astype(np.float32)

        # Weight adaptation
        if not np.all(mu_k == 0.0):
            norm_factor = state.p_x + self.eps
            grad = (E_k[None, :] * np.conj(state.reference_history)) / norm_factor[None, :]
            state.weights += mu_k[None, :] * grad

            if self.leakage > 0.0:
                state.weights *= (1.0 - self.leakage)

            # Time-domain constraint projection: zero second half of impulse response
            w_time = np.real(np.fft.ifft(state.weights, axis=-1)).astype(np.float32)
            w_time[:, B:] = 0.0
            state.weights = np.fft.fft(w_time, axis=-1).astype(np.complex64)

        return e_block, state

    def process(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        state: FDAFState,
    ) -> Tuple[np.ndarray, FDAFState]:
        """Filter a stream of primary and reference microphone samples."""
        d = np.asarray(primary, dtype=np.float32)
        x = np.asarray(reference, dtype=np.float32)

        if d.ndim != 1 or x.ndim != 1:
            raise ValueError("Inputs must be 1-D arrays")
        if len(d) != len(x):
            raise ValueError("primary and reference length mismatch")
        if len(d) == 0:
            return np.empty(0, dtype=np.float32), state

        B = self.block_size
        prev_buf_len = len(state.buf_primary)

        # Concatenate leftover samples from prior calls
        if prev_buf_len > 0:
            all_d = np.concatenate((state.buf_primary, d))
            all_x = np.concatenate((state.buf_reference, x))
        else:
            all_d = d
            all_x = x

        total_samples = len(all_d)
        num_complete_blocks = total_samples // B
        complete_samples = num_complete_blocks * B

        out_blocks = []
        for b_idx in range(num_complete_blocks):
            start = b_idx * B
            end = start + B
            e_b, state = self._process_one_block(all_d[start:end], all_x[start:end], state)
            out_blocks.append(e_b)

        if out_blocks:
            full_completed_out = np.concatenate(out_blocks)
        else:
            full_completed_out = np.empty(0, dtype=np.float32)

        # The first `prev_buf_len` samples of completed output were already emitted
        # during the previous call's partial block handling, so skip them:
        valid_completed_out = full_completed_out[prev_buf_len:]

        # Handle remaining partial block (if total_samples is not a multiple of B)
        rem_len = total_samples - complete_samples
        if rem_len > 0:
            rem_d = all_d[complete_samples:]
            rem_x = all_x[complete_samples:]

            # Partial block overlap-save prediction without weight adaptation
            temp_ref = np.zeros(self.fft_size, dtype=np.float32)
            temp_ref[:B] = state.prev_ref
            temp_ref[B : B + rem_len] = rem_x
            temp_X = np.fft.fft(temp_ref).astype(np.complex64)

            # Estimate with current partition weights
            Y_temp = state.weights[0] * temp_X
            if self.num_partitions > 1:
                Y_temp += np.sum(state.weights[1:] * state.reference_history[:-1], axis=0)

            y_time = np.real(np.fft.ifft(Y_temp)).astype(np.float32)
            y_est_partial = y_time[B : B + rem_len]
            partial_out = rem_d - y_est_partial

            state.buf_primary = rem_d.copy()
            state.buf_reference = rem_x.copy()

            if len(valid_completed_out) > 0:
                output = np.concatenate((valid_completed_out, partial_out))
            else:
                output = partial_out
        else:
            state.buf_primary = np.empty(0, dtype=np.float32)
            state.buf_reference = np.empty(0, dtype=np.float32)
            output = valid_completed_out

        state.warmup_samples += len(d)
        return output, state


# Aliases
ReferenceNoiseFDAF = BattlefieldFDAF
FDAF = BattlefieldFDAF
