import torch

from highspl.training.losses import BattlefieldHighSPLLoss, ERLELoss


def test_erle_speech_weight_uses_smoothed_envelope_not_zero_crossings():
    """The penalty must not switch on at every zero crossing of speech."""
    n = 9600
    t = torch.arange(n) / 48000.0
    speech = torch.zeros(1, n)
    speech[0, : n // 2] = 0.3 * torch.sin(2 * torch.pi * 200 * t[: n // 2])
    residual = torch.full(
        (1, n), 0.1
    )  # a big, constant residual while speech is active
    residual[0, n // 2 :] = 1e-3
    reference = torch.ones(1, n) * 0.1

    smoothed = ERLELoss(mode="residual", smoothing_samples=480)(
        residual, reference, speech_target=speech
    )
    instantaneous = ERLELoss(mode="residual", smoothing_samples=1)(
        residual, reference, speech_target=speech
    )

    assert smoothed < 0.1 * instantaneous


def test_composite_loss_clipping_term_is_off_by_default():
    assert BattlefieldHighSPLLoss().clipping_weight == 0.0
