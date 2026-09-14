import torch

from highspl.training.losses import (
    BattlefieldHighSPLLoss,
    ClippingPenaltyLoss,
    ERLELoss,
    MultiResolutionSTFTLoss,
    SingleResolutionSTFTLoss,
    SISDRLoss,
    SISNRLoss,
)


def test_single_resolution_stft_loss():
    loss_fn = SingleResolutionSTFTLoss(n_fft=512, hop_size=120, win_length=240)
    x = torch.randn(2, 4800, requires_grad=True)
    target = torch.randn(2, 4800)

    sc_loss, mag_loss = loss_fn(x, target)

    assert torch.isfinite(sc_loss)
    assert torch.isfinite(mag_loss)
    assert sc_loss.item() > 0
    assert mag_loss.item() > 0

    total = sc_loss + mag_loss
    total.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_multi_resolution_stft_loss_identical_signals():
    loss_fn = MultiResolutionSTFTLoss(
        n_ffts=[512, 1024],
        hop_sizes=[64, 128],
        win_lengths=[256, 512],
    )
    target = torch.randn(2, 4000)

    loss = loss_fn(target, target)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-4


def test_multi_resolution_stft_loss_different_shapes():
    loss_fn = MultiResolutionSTFTLoss()

    # 1D input (T,)
    sig_1d = torch.randn(4000)
    loss_1d = loss_fn(sig_1d, sig_1d)
    assert torch.isfinite(loss_1d)

    # 2D input (B, T)
    sig_2d = torch.randn(2, 4000)
    loss_2d = loss_fn(sig_2d, sig_2d)
    assert torch.isfinite(loss_2d)

    # 3D input (B, 1, T)
    sig_3d = torch.randn(2, 1, 4000)
    loss_3d = loss_fn(sig_3d, sig_3d)
    assert torch.isfinite(loss_3d)


def test_sisnr_loss_scale_invariance():
    loss_fn = SISNRLoss()
    target = torch.randn(2, 4000)

    # Estimate scaled by a constant factor 3.5
    estimate_scaled = target * 3.5

    loss_identical = loss_fn(target, target)
    loss_scaled = loss_fn(estimate_scaled, target)

    # Both should have very high SI-SNR (negative loss), almost identical
    assert torch.isfinite(loss_identical)
    assert torch.isfinite(loss_scaled)
    assert torch.allclose(loss_identical, loss_scaled, atol=0.05)
    assert loss_identical.item() < -50.0  # SI-SNR > 50 dB

    # With non-zero distortion, scaling the estimate preserves SI-SNR identically
    noise = torch.randn(2, 4000) * 0.1
    estimate = target + noise
    est_scaled = estimate * 3.5
    loss_est = loss_fn(estimate, target)
    loss_est_scaled = loss_fn(est_scaled, target)
    assert torch.allclose(loss_est, loss_est_scaled, atol=0.05)


def test_sisnr_loss_degradation_with_noise():
    loss_fn = SISNRLoss()
    target = torch.randn(2, 4000)

    clean_loss = loss_fn(target, target)
    noisy_loss = loss_fn(torch.randn(2, 4000), target)

    # Loss is -SI-SNR, so noisy estimate should have higher (worse) loss
    assert noisy_loss.item() > clean_loss.item()


def test_sisnr_loss_backward():
    loss_fn = SISNRLoss()
    estimate = torch.randn(2, 2000, requires_grad=True)
    target = torch.randn(2, 2000)

    loss = loss_fn(estimate, target)
    loss.backward()

    assert estimate.grad is not None
    assert torch.all(torch.isfinite(estimate.grad))


def test_sisdr_alias():
    sisnr = SISNRLoss()
    sisdr = SISDRLoss()

    est = torch.randn(2, 1600)
    tgt = torch.randn(2, 1600)

    assert torch.equal(sisnr(est, tgt), sisdr(est, tgt))


def test_erle_loss_ratio_and_db_modes():
    erle_ratio = ERLELoss(mode="ratio")
    erle_db = ERLELoss(mode="db")

    reference = torch.randn(2, 4000)
    # Residual with significant cancellation (20 dB attenuated)
    residual_quiet = reference * 0.05
    # Residual without cancellation
    residual_loud = reference * 1.0

    loss_quiet_ratio = erle_ratio(residual_quiet, reference)
    loss_loud_ratio = erle_ratio(residual_loud, reference)
    assert loss_quiet_ratio.item() < loss_loud_ratio.item()
    assert loss_quiet_ratio.item() >= 0.0

    loss_quiet_db = erle_db(residual_quiet, reference)
    loss_loud_db = erle_db(residual_loud, reference)
    assert loss_quiet_db.item() < loss_loud_db.item()


def test_erle_loss_with_speech_mask():
    erle_loss = ERLELoss(mode="ratio")

    reference = torch.ones(1, 1000)
    _residual = torch.ones(1, 1000)

    # Speech mask: speech is present in first 500 samples, absent in next 500
    speech_mask = torch.zeros(1, 1000)
    speech_mask[:, :500] = 1.0

    # If residual only has energy during active speech, non-speech residual is zero
    residual_only_in_speech = torch.zeros(1, 1000)
    residual_only_in_speech[:, :500] = 1.0

    loss = erle_loss(residual_only_in_speech, reference, speech_mask=speech_mask)
    assert loss.item() < 1e-4


def test_clipping_penalty_loss():
    clipping_loss_l2 = ClippingPenaltyLoss(threshold=0.99, penalty_type="l2")
    clipping_loss_l1 = ClippingPenaltyLoss(threshold=0.99, penalty_type="l1")

    # Clean signal strictly within [-0.95, 0.95]
    clean_signal = torch.linspace(-0.95, 0.95, 1000)
    assert clipping_loss_l2(clean_signal).item() == 0.0
    assert clipping_loss_l1(clean_signal).item() == 0.0

    # Signal with clipping peaks exceeding 0.99
    clipped_signal = clean_signal.clone()
    clipped_signal[100:150] = 1.5
    clipped_signal[200:250] = -1.2

    loss_l2 = clipping_loss_l2(clipped_signal)
    loss_l1 = clipping_loss_l1(clipped_signal)

    assert loss_l2.item() > 0.0
    assert loss_l1.item() > 0.0

    # Gradient check
    clipped_signal.requires_grad_(True)
    loss = clipping_loss_l2(clipped_signal)
    loss.backward()
    assert clipped_signal.grad is not None
    assert torch.all(torch.isfinite(clipped_signal.grad))


def test_battlefield_high_spl_composite_loss():
    loss_fn = BattlefieldHighSPLLoss(
        stft_weight=1.0,
        sisnr_weight=0.5,
        erle_weight=0.2,
        clipping_weight=5.0,
    )

    estimate = torch.randn(2, 3200, requires_grad=True)
    target = torch.randn(2, 3200)
    reference = torch.randn(2, 3200)
    mask = torch.zeros(2, 3200)

    total_loss, components = loss_fn(
        estimate,
        target,
        reference=reference,
        speech_mask=mask,
        return_components=True,
    )

    assert torch.isfinite(total_loss)
    assert "total_loss" in components
    assert "stft_loss" in components
    assert "sisnr_loss" in components
    assert "erle_loss" in components
    assert "clipping_loss" in components

    total_loss.backward()
    assert estimate.grad is not None
    assert torch.all(torch.isfinite(estimate.grad))


def test_battlefield_high_spl_zero_weight_disables_component():
    loss_fn = BattlefieldHighSPLLoss(
        stft_weight=0.0,
        sisnr_weight=1.0,
        erle_weight=0.0,
        clipping_weight=0.0,
    )

    est = torch.randn(1, 2000)
    tgt = torch.randn(1, 2000)

    total_loss, components = loss_fn(est, tgt, return_components=True)

    assert torch.isclose(total_loss, components["sisnr_loss"])
    assert components["stft_loss"].item() == 0.0
    assert components["erle_loss"].item() == 0.0
    assert components["clipping_loss"].item() == 0.0
