from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _align_shape(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in (batch_size, time_samples) shape."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    elif tensor.dim() == 3:
        if tensor.size(1) == 1:
            return tensor.squeeze(1)
        elif tensor.size(2) == 1:
            return tensor.squeeze(2)
        else:
            # Flatten channels into batch dimension
            return tensor.reshape(-1, tensor.size(-1))
    elif tensor.dim() == 2:
        return tensor
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D tensor, got shape {tuple(tensor.shape)}")


class SingleResolutionSTFTLoss(nn.Module):
    """
    Spectral convergence and log STFT magnitude loss for a single resolution.

    L_sc = || |Y| - |Y_hat| ||_F / (|| |Y| ||_F + eps)
    L_mag = (1 / M) * || log(|Y| + eps) - log(|Y_hat| + eps) ||_1
    """

    def __init__(
        self,
        n_fft: int,
        hop_size: int,
        win_length: int,
        window_fn=torch.hann_window,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_length = win_length
        self.eps = eps
        self.register_buffer("window", window_fn(win_length))

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude safely with numerical stability."""
        window = self.window.to(dtype=x.dtype, device=x.device)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        # Safe magnitude calculation to avoid NaN gradients at zero
        mag = torch.sqrt(torch.clamp(spec.real**2 + spec.imag**2, min=self.eps))
        return mag

    def forward(
        self, estimate: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        estimate_2d = _align_shape(estimate)
        target_2d = _align_shape(target)

        est_mag = self._stft_magnitude(estimate_2d)
        tgt_mag = self._stft_magnitude(target_2d)

        # Spectral convergence loss
        norm_diff = torch.norm(tgt_mag - est_mag, p="fro", dim=(-2, -1))
        norm_tgt = torch.norm(tgt_mag, p="fro", dim=(-2, -1))
        sc_loss = torch.mean(norm_diff / (norm_tgt + self.eps))

        # Log magnitude loss
        log_tgt = torch.log(tgt_mag + self.eps)
        log_est = torch.log(est_mag + self.eps)
        mag_loss = F.l1_loss(log_est, log_tgt)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss combining spectral convergence and log magnitude losses
    over multiple FFT resolutions.

    Default resolutions:
        - n_ffts: [512, 1024, 2048]
        - hop_sizes: [50, 120, 240]
        - win_lengths: [240, 600, 1200]
    """

    def __init__(
        self,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (50, 120, 240),
        win_lengths: Sequence[int] = (240, 600, 1200),
        factor_sc: float = 1.0,
        factor_mag: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not (len(n_ffts) == len(hop_sizes) == len(win_lengths)):
            raise ValueError("n_ffts, hop_sizes, and win_lengths must have the same length")

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.reduction = reduction
        self.eps = eps

        self.loss_layers = nn.ModuleList([
            SingleResolutionSTFTLoss(n_fft, hop, win, eps=eps)
            for n_fft, hop, win in zip(n_ffts, hop_sizes, win_lengths)
        ])

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        total_sc = estimate.new_zeros(1).squeeze()
        total_mag = estimate.new_zeros(1).squeeze()

        for layer in self.loss_layers:
            sc_l, mag_l = layer(estimate, target)
            total_sc = total_sc + sc_l
            total_mag = total_mag + mag_l

        if self.reduction == "mean":
            n = float(len(self.loss_layers))
            total_sc = total_sc / n
            total_mag = total_mag / n

        return self.factor_sc * total_sc + self.factor_mag * total_mag


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR / SI-SDR) Loss.
    Returns negative SI-SNR (in dB) to be minimized during gradient descent.

    Formula:
        s_target = (<x_hat, x> / (||x||^2 + eps)) * x
        e_noise = x_hat - s_target
        SI-SNR = 10 * log10(||s_target||^2 / (||e_noise||^2 + eps))
        Loss = -mean(SI-SNR)
    """

    def __init__(
        self,
        zero_mean: bool = True,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.zero_mean = zero_mean
        self.reduction = reduction
        self.eps = eps

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        est = _align_shape(estimate)
        tgt = _align_shape(target)

        if est.shape != tgt.shape:
            raise ValueError(f"Shape mismatch: estimate {tuple(est.shape)} vs target {tuple(tgt.shape)}")

        if self.zero_mean:
            est = est - torch.mean(est, dim=-1, keepdim=True)
            tgt = tgt - torch.mean(tgt, dim=-1, keepdim=True)

        dot = torch.sum(est * tgt, dim=-1, keepdim=True)
        tgt_energy = torch.sum(tgt**2, dim=-1, keepdim=True)

        # Scale factor projection
        alpha = dot / (tgt_energy + self.eps)
        s_target = alpha * tgt
        e_noise = est - s_target

        target_power = torch.sum(tgt**2, dim=-1)
        noise_power = torch.sum((e_noise / (torch.abs(alpha) + self.eps))**2, dim=-1)

        si_snr = 10.0 * torch.log10((target_power + self.eps) / (noise_power + self.eps))

        if self.reduction == "mean":
            return -torch.mean(si_snr)
        elif self.reduction == "sum":
            return -torch.sum(si_snr)
        elif self.reduction == "none":
            return -si_snr
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


# Alias SISDRLoss for SI-SNR Loss
SISDRLoss = SISNRLoss


class ERLELoss(nn.Module):
    """
    Echo Return Loss Enhancement (ERLE) penalty loss.
    Penalizes residual energy when near-end speech is absent or low.

    ERLE = 10 * log10(P_reference / P_residual)

    When mode == "ratio":
        loss = P_residual_non_speech / (P_reference_non_speech + eps)
    When mode == "db":
        loss = -ERLE = 10 * log10((P_residual + eps) / (P_reference + eps))
    """

    def __init__(
        self,
        mode: str = "ratio",
        speech_threshold: float = 1e-4,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if mode not in ("ratio", "db", "residual"):
            raise ValueError(f"mode must be one of ('ratio', 'db', 'residual'), got {mode}")
        self.mode = mode
        self.speech_threshold = speech_threshold
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        residual: torch.Tensor,
        reference: torch.Tensor,
        speech_target: torch.Tensor | None = None,
        speech_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        res = _align_shape(residual)
        ref = _align_shape(reference)

        min_len = min(res.size(-1), ref.size(-1))
        res = res[..., :min_len]
        ref = ref[..., :min_len]

        # Determine non-speech weighting (1.0 = purely echo/noise, 0.0 = active speech)
        if speech_mask is not None:
            mask = _align_shape(speech_mask)[..., :min_len]
            non_speech_weight = torch.clamp(1.0 - mask, 0.0, 1.0)
        elif speech_target is not None:
            sp = _align_shape(speech_target)[..., :min_len]
            speech_power = sp**2
            # Soft weighting: 1 when speech power << speech_threshold, 0 when speech power >> speech_threshold
            non_speech_weight = torch.sigmoid(-10.0 * (speech_power / (self.speech_threshold + self.eps) - 1.0))
        else:
            non_speech_weight = torch.ones_like(res)

        weight_sum = torch.sum(non_speech_weight, dim=-1) + self.eps

        p_res = torch.sum(non_speech_weight * (res**2), dim=-1) / weight_sum
        p_ref = torch.sum(non_speech_weight * (ref**2), dim=-1) / weight_sum

        if self.mode == "ratio":
            loss = p_res / (p_ref + self.eps)
        elif self.mode == "db":
            loss = 10.0 * torch.log10((p_res + self.eps) / (p_ref + self.eps))
        elif self.mode == "residual":
            loss = p_res
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class ClippingPenaltyLoss(nn.Module):
    """
    Penalizes signal amplitudes that exceed the linear dynamic range (|x| > threshold).
    Designed to prevent digital clipping in high-SPL environments (> 0.99 amplitude).
    """

    def __init__(
        self,
        threshold: float = 0.99,
        penalty_type: str = "l2",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        if penalty_type not in ("l1", "l2"):
            raise ValueError(f"penalty_type must be 'l1' or 'l2', got {penalty_type}")
        self.penalty_type = penalty_type
        self.reduction = reduction

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        excess = F.relu(torch.abs(signal) - self.threshold)

        if self.penalty_type == "l2":
            penalty = excess**2
        else:
            penalty = excess

        if self.reduction == "mean":
            return torch.mean(penalty)
        elif self.reduction == "sum":
            return torch.sum(penalty)
        elif self.reduction == "none":
            return penalty
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")


class BattlefieldHighSPLLoss(nn.Module):
    """
    Composite loss combining:
      1. Multi-resolution STFT loss (spectral convergence + log magnitude)
      2. Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss
      3. ERLE penalty (echo/noise residual energy reduction)
      4. Dynamic range clipping penalty (|x| > 0.99)

    All weights are fully configurable.
    """

    def __init__(
        self,
        stft_weight: float = 1.0,
        sisnr_weight: float = 1.0,
        erle_weight: float = 0.5,
        clipping_weight: float = 10.0,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Sequence[int] = (50, 120, 240),
        win_lengths: Sequence[int] = (240, 600, 1200),
        clipping_threshold: float = 0.99,
        erle_mode: str = "ratio",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.stft_weight = float(stft_weight)
        self.sisnr_weight = float(sisnr_weight)
        self.erle_weight = float(erle_weight)
        self.clipping_weight = float(clipping_weight)

        self.stft_loss = MultiResolutionSTFTLoss(
            n_ffts=n_ffts,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            eps=eps,
        )
        self.sisnr_loss = SISNRLoss(eps=eps)
        self.erle_loss = ERLELoss(mode=erle_mode, eps=eps)
        self.clipping_loss = ClippingPenaltyLoss(threshold=clipping_threshold)

        self.last_components: dict[str, torch.Tensor] = {}

    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor | None = None,
        speech_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute composite battlefield loss.

        Args:
            estimate: Model output signal (B, T) or (B, 1, T) or (T,)
            target: Ground truth clean speech signal (B, T) or (B, 1, T) or (T,)
            reference: Optional far-end echo/noise reference or raw mixture (for ERLE penalty)
            speech_mask: Optional near-end speech activity mask (1 = speech, 0 = non-speech)
            return_components: If True, return (total_loss, components_dict)
        """
        zero = estimate.new_zeros(1).squeeze()

        # 1. Multi-resolution STFT loss
        if self.stft_weight > 0:
            stft_val = self.stft_loss(estimate, target)
        else:
            stft_val = zero

        # 2. SI-SNR loss
        if self.sisnr_weight > 0:
            sisnr_val = self.sisnr_loss(estimate, target)
        else:
            sisnr_val = zero

        # 3. ERLE loss
        if self.erle_weight > 0 and reference is not None:
            erle_val = self.erle_loss(
                residual=estimate,
                reference=reference,
                speech_target=target,
                speech_mask=speech_mask,
            )
        else:
            erle_val = zero

        # 4. Clipping penalty loss
        if self.clipping_weight > 0:
            clipping_val = self.clipping_loss(estimate)
        else:
            clipping_val = zero

        total_loss = (
            self.stft_weight * stft_val
            + self.sisnr_weight * sisnr_val
            + self.erle_weight * erle_val
            + self.clipping_weight * clipping_val
        )

        components = {
            "total_loss": total_loss,
            "stft_loss": stft_val,
            "sisnr_loss": sisnr_val,
            "erle_loss": erle_val,
            "clipping_loss": clipping_val,
        }
        self.last_components = components

        if return_components:
            return total_loss, components
        return total_loss
