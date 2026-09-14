import numpy as np
import pytest

from highspl.dsp.nlms import NLMS, VSSNLMS, NLMSState, RobustNLMS, VSSNLMSState


def test_reset_creates_correct_state():
    nlms = RobustNLMS(filter_length=32)

    state = nlms.reset()

    assert isinstance(state, NLMSState)
    assert state.weights.shape == (32,)
    assert state.x_hist.shape == (32,)
    assert np.all(state.weights == 0.0)
    assert np.all(state.x_hist == 0.0)
    assert state.energy_ema > 0.0
    assert state.warmup_samples == 0


def test_zero_reference_is_numerically_stable():
    nlms = RobustNLMS(filter_length=32)
    state = nlms.reset()

    primary = np.ones(256, dtype=np.float32)
    reference = np.zeros(256, dtype=np.float32)

    output, state = nlms.process(primary, reference, state)

    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(state.weights))
    assert np.allclose(output, primary)
    assert state.warmup_samples == 256


def test_primary_reference_length_mismatch():
    nlms = RobustNLMS()
    state = nlms.reset()

    primary = np.zeros(100, dtype=np.float32)
    reference = np.zeros(99, dtype=np.float32)

    with pytest.raises(ValueError, match="length mismatch"):
        nlms.process(primary, reference, state)


def test_non_1d_input_rejected():
    nlms = RobustNLMS()
    state = nlms.reset()

    primary = np.zeros((2, 100), dtype=np.float32)
    reference = np.zeros((2, 100), dtype=np.float32)

    with pytest.raises(ValueError, match="1-D"):
        nlms.process(primary, reference, state)


def test_state_continuity_across_chunks():
    rng = np.random.default_rng(1337)

    nlms = RobustNLMS(filter_length=16)

    primary = rng.normal(size=1000).astype(np.float32)
    reference = rng.normal(size=1000).astype(np.float32)

    state_full = nlms.reset()
    full_output, full_state = nlms.process(primary, reference, state_full)

    state_chunked = nlms.reset()

    out_a, state_chunked = nlms.process(
        primary[:400],
        reference[:400],
        state_chunked,
    )
    out_b, state_chunked = nlms.process(
        primary[400:],
        reference[400:],
        state_chunked,
    )

    chunked_output = np.concatenate([out_a, out_b])

    assert np.allclose(chunked_output, full_output, atol=1e-6)
    assert np.allclose(state_chunked.weights, full_state.weights, atol=1e-6)
    assert np.allclose(state_chunked.x_hist, full_state.x_hist, atol=1e-6)
    assert state_chunked.warmup_samples == 1000


def test_impulse_freezes_adaptation():
    nlms = RobustNLMS(
        filter_length=8,
        step_size=0.25,
        freeze_ratio_db=12.0,
    )
    state = nlms.reset()

    rng = np.random.default_rng(42)

    reference = rng.normal(0.0, 0.01, 100).astype(np.float32)
    primary = reference.copy()

    _, state = nlms.process(primary, reference, state)

    weights_before_impulse = state.weights.copy()

    reference_impulse = np.zeros(20, dtype=np.float32)
    primary_impulse = np.ones(20, dtype=np.float32)

    reference_impulse[10] = 100.0

    _, state = nlms.process(
        primary_impulse,
        reference_impulse,
        state,
    )

    weights_after_impulse = state.weights.copy()

    assert np.all(np.isfinite(weights_after_impulse))
    assert np.allclose(
        weights_after_impulse,
        weights_before_impulse,
        atol=1e-4,
    )


def test_correlated_noise_is_reduced():
    rng = np.random.default_rng(7)

    nlms = RobustNLMS(
        filter_length=32,
        step_size=0.5,
        leakage=1e-5,
    )
    state = nlms.reset()

    reference = rng.normal(0.0, 0.5, 5000).astype(np.float32)

    speech = rng.normal(0.0, 0.05, 5000).astype(np.float32)
    noise = reference * 0.8

    primary = speech + noise

    output, state = nlms.process(primary, reference, state)

    input_error = np.mean((primary[1000:] - speech[1000:]) ** 2)
    output_error = np.mean((output[1000:] - speech[1000:]) ** 2)

    assert np.isfinite(input_error)
    assert np.isfinite(output_error)
    assert output_error < input_error
    assert np.all(np.isfinite(state.weights))


def test_canonical_nlms_validation():
    nlms = NLMS(filter_length=32)
    state = nlms.reset()

    assert isinstance(state, NLMSState)
    assert state.weights.shape == (32,)

    # Length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        nlms.process(np.zeros(10), np.zeros(9), state)

    # 1-D check
    with pytest.raises(ValueError, match="1-D"):
        nlms.process(np.zeros((2, 10)), np.zeros((2, 10)), state)


def test_canonical_nlms_reduces_noise():
    rng = np.random.default_rng(7)
    nlms = NLMS(filter_length=32, step_size=0.5)
    state = nlms.reset()

    reference = rng.normal(0.0, 0.5, 5000).astype(np.float32)
    speech = rng.normal(0.0, 0.05, 5000).astype(np.float32)
    noise = reference * 0.8
    primary = speech + noise

    output, state = nlms.process(primary, reference, state)

    input_error = np.mean((primary[1000:] - speech[1000:]) ** 2)
    output_error = np.mean((output[1000:] - speech[1000:]) ** 2)

    assert output_error < input_error


def test_vss_nlms_validation_and_reduction():
    vss = VSSNLMS(filter_length=32)
    state = vss.reset()

    assert isinstance(state, VSSNLMSState)

    # Length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        vss.process(np.zeros(10), np.zeros(9), state)

    # 1-D check
    with pytest.raises(ValueError, match="1-D"):
        vss.process(np.zeros((2, 10)), np.zeros((2, 10)), state)

    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 0.5, 5000).astype(np.float32)
    speech = rng.normal(0.0, 0.05, 5000).astype(np.float32)
    noise = reference * 0.8
    primary = speech + noise

    output, state = vss.process(primary, reference, state)

    input_error = np.mean((primary[1000:] - speech[1000:]) ** 2)
    output_error = np.mean((output[1000:] - speech[1000:]) ** 2)

    assert output_error < input_error
