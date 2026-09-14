import numpy as np
import pytest

from highspl.dsp.fdaf import FDAF, BattlefieldFDAF, FDAFState, ReferenceNoiseFDAF


def test_reset_creates_correct_state():
    fdaf = BattlefieldFDAF(block_size=128, num_partitions=4)
    state = fdaf.reset()

    assert isinstance(state, FDAFState)
    assert state.weights.shape == (4, 256)
    assert state.reference_history.shape == (4, 256)
    assert state.prev_ref.shape == (128,)
    assert np.all(state.weights == 0.0)
    assert np.all(state.reference_history == 0.0)
    assert np.all(state.prev_ref == 0.0)
    assert state.warmup_samples == 0


def test_aliases():
    assert FDAF is BattlefieldFDAF
    assert ReferenceNoiseFDAF is BattlefieldFDAF


def test_primary_reference_length_mismatch():
    fdaf = FDAF(block_size=128)
    state = fdaf.reset()

    primary = np.zeros(256, dtype=np.float32)
    reference = np.zeros(255, dtype=np.float32)

    with pytest.raises(ValueError, match="length mismatch"):
        fdaf.process(primary, reference, state)


def test_non_1d_input_rejected():
    fdaf = FDAF(block_size=128)
    state = fdaf.reset()

    primary = np.zeros((2, 128), dtype=np.float32)
    reference = np.zeros((2, 128), dtype=np.float32)

    with pytest.raises(ValueError, match="1-D"):
        fdaf.process(primary, reference, state)


def test_zero_reference_is_numerically_stable():
    fdaf = FDAF(block_size=128)
    state = fdaf.reset()

    primary = np.ones(512, dtype=np.float32)
    reference = np.zeros(512, dtype=np.float32)

    output, state = fdaf.process(primary, reference, state)

    assert np.all(np.isfinite(output))
    assert np.all(np.isfinite(state.weights))
    assert np.allclose(output, primary)
    assert state.warmup_samples == 512


def test_correlated_noise_is_reduced():
    rng = np.random.default_rng(42)

    fdaf = BattlefieldFDAF(
        block_size=128,
        num_partitions=4,
        step_size=0.1,
    )
    state = fdaf.reset()

    reference = rng.normal(0.0, 0.5, 5120).astype(np.float32)
    speech = rng.normal(0.0, 0.05, 5120).astype(np.float32)
    noise = reference * 0.7

    primary = speech + noise

    output, state = fdaf.process(primary, reference, state)

    assert len(output) == len(primary)
    input_error = np.mean((primary[1024:] - speech[1024:]) ** 2)
    output_error = np.mean((output[1024:] - speech[1024:]) ** 2)

    assert np.isfinite(input_error)
    assert np.isfinite(output_error)
    assert output_error < input_error
    assert np.all(np.isfinite(state.weights))


def test_clipping_protection_freezes_adaptation():
    fdaf = BattlefieldFDAF(
        block_size=128,
        num_partitions=2,
        step_size=0.1,
        clip_threshold=0.95,
    )
    state = fdaf.reset()
    rng = np.random.default_rng(123)

    # Initial adaptation
    ref = rng.normal(0.0, 0.1, 256).astype(np.float32)
    prim = ref.copy()
    _, state = fdaf.process(prim, ref, state)

    weights_before = state.weights.copy()

    # Clipped block
    ref_clipped = np.full(128, 1.5, dtype=np.float32)
    prim_clipped = np.full(128, 1.5, dtype=np.float32)

    _, state = fdaf.process(prim_clipped, ref_clipped, state)
    weights_after = state.weights.copy()

    assert np.allclose(weights_before, weights_after)


def test_time_domain_constraint_projection():
    fdaf = BattlefieldFDAF(
        block_size=128,
        num_partitions=2,
        step_size=0.1,
    )
    state = fdaf.reset()
    rng = np.random.default_rng(999)

    ref = rng.normal(0.0, 0.3, 512).astype(np.float32)
    prim = ref * 0.5
    _, state = fdaf.process(prim, ref, state)

    # Check that second half of each partition impulse response is strictly zero
    w_time = np.real(np.fft.ifft(state.weights, axis=-1))
    for p in range(fdaf.num_partitions):
        assert np.allclose(w_time[p, fdaf.block_size:], 0.0, atol=1e-5)


def test_state_continuity_across_chunks():
    rng = np.random.default_rng(42)

    fdaf = BattlefieldFDAF(block_size=128, num_partitions=2, step_size=0.05)

    primary = rng.normal(size=1024).astype(np.float32)
    reference = rng.normal(size=1024).astype(np.float32)

    state_full = fdaf.reset()
    full_output, full_state = fdaf.process(primary, reference, state_full)

    state_chunked = fdaf.reset()
    out_a, state_chunked = fdaf.process(primary[:512], reference[:512], state_chunked)
    out_b, state_chunked = fdaf.process(primary[512:], reference[512:], state_chunked)

    chunked_output = np.concatenate([out_a, out_b])

    assert np.allclose(chunked_output, full_output, atol=1e-5)
    assert np.allclose(state_chunked.weights, full_state.weights, atol=1e-5)
    assert np.allclose(state_chunked.prev_ref, full_state.prev_ref, atol=1e-5)
    assert state_chunked.warmup_samples == 1024
