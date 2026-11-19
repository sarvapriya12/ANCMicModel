import numpy as np
import pytest

from highspl.data import clipping, mixture, noise_synth, rir, speech_synth

SR = 48_000


# ------------------------------------------------------------------ speech


def test_speech_is_deterministic_and_finite():
    a = speech_synth.synthesize_speech(3.0, np.random.default_rng(5))
    b = speech_synth.synthesize_speech(3.0, np.random.default_rng(5))

    assert np.isfinite(a.wave).all()
    np.testing.assert_array_equal(a.wave, b.wave)


def test_speech_labels_are_consistent():
    clip = speech_synth.synthesize_speech(6.0, np.random.default_rng(1))
    lab = clip.labels

    assert clip.wave.shape == (6 * SR,)
    assert len(lab["f0_hz"]) == len(lab["voiced"]) == len(lab["active"])
    assert lab["f0_hz"][~lab["voiced"]].max() == 0.0
    assert 0.45 < lab["active"].mean() < 0.85  # 15-55 % pauses
    # voiced frames are speech-active frames
    assert lab["active"][lab["voiced"]].all()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_speech_f0_within_human_range(seed):
    clip = speech_synth.synthesize_speech(5.0, np.random.default_rng(seed))
    f0 = clip.labels["f0_hz"][clip.labels["voiced"]]

    assert 80.0 <= f0.min() and f0.max() <= 450.0
    ranges = {"male": (85.0, 170.0), "female": (160.0, 320.0)}
    lo, hi = ranges[clip.labels["voice"]["sex"]]
    assert lo <= np.median(f0) <= hi


def test_speech_formants_follow_vowel_table():
    clip = speech_synth.synthesize_speech(5.0, np.random.default_rng(2))
    f = clip.labels["formants_hz"][clip.labels["voiced"]]

    assert (f[:, 0] > 200).all() and (f[:, 0] < 1100).all()  # F1
    assert (f[:, 1] > 700).all() and (f[:, 1] < 3200).all()  # F2


def test_lombard_raises_f0_and_flattens_tilt():
    rng = np.random.default_rng(11)
    normal = speech_synth.random_voice(rng, 0.0)
    loud = speech_synth.random_voice(np.random.default_rng(11), 1.0)

    assert loud.f0_mean > normal.f0_mean
    assert loud.glottal_fc > normal.glottal_fc


def test_active_speech_rms_ignores_silence():
    rng = np.random.default_rng(0)
    tone = 0.1 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
    padded = np.concatenate(
        [tone, np.zeros(3 * SR) + 1e-6 * rng.standard_normal(3 * SR)]
    )

    assert speech_synth.active_speech_rms(padded, SR) == pytest.approx(
        0.1 / np.sqrt(2), rel=0.05
    )


# ------------------------------------------------------------------- noise


@pytest.mark.parametrize("cls", noise_synth.ALL_CLASSES)
def test_noise_classes_normalised_and_deterministic(cls):
    n = SR * 2
    a = noise_synth.make_noise(cls, n, SR, np.random.default_rng(3))
    b = noise_synth.make_noise(cls, n, SR, np.random.default_rng(3))

    np.testing.assert_array_equal(a.signal, b.signal)
    assert np.isfinite(a.signal).all()
    if a.kind == noise_synth.STATIONARY:
        assert np.sqrt(np.mean(a.signal**2)) == pytest.approx(1.0, rel=1e-6)
    else:
        assert np.abs(a.signal).max() == pytest.approx(1.0, rel=1e-6)


def _band_fraction(x, lo, hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return spec[(f >= lo) & (f < hi)].sum() / spec.sum()


def test_noise_spectra_match_physics():
    n = SR * 4
    tank = noise_synth.make_noise("tank", n, SR, np.random.default_rng(1)).signal
    artillery = noise_synth.make_noise(
        "artillery", n, SR, np.random.default_rng(1)
    ).signal
    gun = noise_synth.make_noise("gunfire", n, SR, np.random.default_rng(1)).signal
    jet = noise_synth.make_noise("jet", n, SR, np.random.default_rng(1)).signal

    assert _band_fraction(tank, 0, 1000) > 0.75  # tracks + diesel are LF
    assert _band_fraction(artillery, 0, 300) > 0.8  # distance low-pass
    assert 0.02 < _band_fraction(gun, 1000, 6000)  # blast + crack reach the kHz
    assert _band_fraction(jet, 1000, 6000) > 0.2  # tonal fan / whine in-band


def test_gunfire_is_impulsive_and_wind_is_incoherent():
    n = SR * 3
    gun = noise_synth.make_noise("gunfire", n, SR, np.random.default_rng(4))
    assert gun.kind == noise_synth.IMPULSIVE
    assert np.abs(gun.signal).max() / np.sqrt(np.mean(gun.signal**2)) > 8.0  # crest

    w1 = noise_synth.make_noise("wind", n, SR, np.random.default_rng(1)).signal
    w2 = noise_synth.make_noise("wind", n, SR, np.random.default_rng(2)).signal
    assert abs(np.corrcoef(w1, w2)[0, 1]) < 0.05


def test_unknown_noise_class_rejected():
    with pytest.raises(ValueError, match="unknown noise class"):
        noise_synth.make_noise("nope", 100, SR, np.random.default_rng(0))


# --------------------------------------------------------------------- rir


def test_fractional_delay_is_exact_for_band_limited_signals():
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 1000 * t)
    y = rir.fractional_delay(x, 2.4)

    expected = np.sin(2 * np.pi * 1000 * (t - 2.4 / SR))
    np.testing.assert_allclose(y[500:-500], expected[500:-500], atol=2e-3)


def test_plane_wave_delay_sign_and_size():
    geom = rir.PodGeometry(
        spacing_m=0.0343, shadow_db=0.0
    )  # d/c = 0.1 ms = 4.8 samples
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(SR // 2)
    sig = np.convolve(sig, np.ones(8) / 8, mode="same")

    def lag(theta):
        f, r = rir.plane_wave_to_mics(sig, theta, geom, SR)
        c = np.correlate(r, f, "full")
        return np.argmax(c) - (len(f) - 1)

    assert lag(0.0) == 5  # in front: rear mic hears it later
    assert lag(180.0) == -5  # behind: rear mic hears it first
    assert lag(90.0) == 0


def test_mouth_is_louder_at_front_by_spherical_spreading():
    geom = rir.PodGeometry(spacing_m=0.015, mouth_m=0.05, shadow_db=0.0)
    x = np.random.default_rng(0).standard_normal(SR)
    f, r = rir.mouth_to_mics(x, geom, SR)

    ratio_db = 20 * np.log10(np.std(f) / np.std(r))
    assert ratio_db == pytest.approx(20 * np.log10((0.05 + 0.015) / 0.05), abs=0.1)


def test_diffuse_pair_has_sinc_coherence():
    rng = np.random.default_rng(7)
    n = SR * 20
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    d = 0.02
    x, y = rir.diffuse_pair(a, b, d, SR)

    from scipy import signal as sps

    f, coh = sps.coherence(x, y, fs=SR, nperseg=1024)
    expected = rir.diffuse_coherence(f, d) ** 2
    mask = (f > 200) & (f < 8000)
    assert np.mean(np.abs(coh[mask] - expected[mask])) < 0.05


# ---------------------------------------------------------------- hardware


def test_aop_maps_to_full_scale():
    hw = clipping.MicHardware(aop_db=120.0)
    amp = np.sqrt(2.0) * clipping.spl_to_pa_rms(120.0)
    t = np.arange(SR // 2) / SR
    x = amp * np.sin(2 * np.pi * 1000 * t)

    y = apply_linear = clipping.apply_linear_chain(x, hw, SR)
    del apply_linear
    assert np.abs(y[2000:]).max() == pytest.approx(1.0, abs=0.05)


def test_overload_clips_and_clean_signal_does_not():
    rng = np.random.default_rng(0)
    hw = clipping.MicHardware(aop_db=120.0, self_noise_db_spl=0.0)
    loud = clipping.spl_to_pa_rms(135.0) * rng.standard_normal(SR)
    quiet = clipping.spl_to_pa_rms(100.0) * rng.standard_normal(SR)

    _, _, info_loud = clipping.apply_chain(loud, loud, hw, SR, rng)
    _, _, info_quiet = clipping.apply_chain(quiet, quiet, hw, SR, rng)

    assert info_loud["clip_fraction_front"] > 0.5
    assert info_quiet["clip_fraction_front"] < 0.01


def test_output_is_on_the_24bit_grid_and_bounded():
    rng = np.random.default_rng(1)
    hw = clipping.MicHardware()
    x = clipping.spl_to_pa_rms(110.0) * rng.standard_normal(SR // 4)

    f, r, _ = clipping.apply_chain(x, x, hw, SR, rng)

    scaled = f.astype(np.float64) * 2**23
    np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-3)
    assert np.abs(f).max() <= 1.0 and np.abs(r).max() <= 1.0


def test_no_microcontroller_stages_by_default():
    hw = clipping.MicHardware()

    assert hw.dc_blocker == 0.0  # no firmware DC blocker on a Pi-direct build
    assert hw.bits == 24  # I2S data is 24-bit, not an ESP32 16-bit stream
    assert hw.dropout_prob == 0.0


def test_alsa_overrun_zeroes_both_channels_together():
    rng = np.random.default_rng(2)
    hw = clipping.MicHardware(dropout_prob=0.3, dropout_period_samples=128)
    x = clipping.spl_to_pa_rms(100.0) * rng.standard_normal(SR // 2)

    f, r, info = clipping.apply_chain(x, x, hw, SR, rng)

    lost = info["lost_mask"]
    assert lost.any()
    assert (f[lost] == 0).all() and (r[lost] == 0).all()
    assert info["lost_fraction"] == pytest.approx(lost.mean())


def test_i2s_offset_delays_only_the_reference():
    rng = np.random.default_rng(3)
    hw = clipping.MicHardware(i2s_offset_samples=0.0, self_noise_db_spl=-20.0)
    hw_off = clipping.MicHardware(i2s_offset_samples=1.0, self_noise_db_spl=-20.0)
    x = clipping.spl_to_pa_rms(100.0) * np.convolve(
        rng.standard_normal(SR // 2), np.ones(6) / 6, mode="same"
    )
    f0, r0, _ = clipping.apply_chain(x, x, hw, SR, np.random.default_rng(9))
    f1, r1, _ = clipping.apply_chain(x, x, hw_off, SR, np.random.default_rng(9))

    np.testing.assert_allclose(f0, f1, atol=2e-4)
    c = np.correlate(r1.astype(np.float64), r0.astype(np.float64), "full")
    assert np.argmax(c) - (len(r0) - 1) == 1


def test_drift_resample_changes_length_by_ppm():
    x = np.random.default_rng(0).standard_normal(1_000_000)
    y = clipping.drift_resample(x, 50.0)

    assert abs(len(y) - len(x) * (1 - 50e-6)) <= 2


# ----------------------------------------------------------------- mixture


def _spec(index=0, tier="S1", duration=2.0, **kw):
    return mixture.sample_scene_spec(5, index, duration_s=duration, tier=tier, **kw)


def test_scene_spec_is_deterministic_and_stratified():
    assert _spec(7).to_json() == _spec(7).to_json()

    groups = {mixture.sample_scene_spec(1, i).group for i in range(6)}
    bands = {mixture.sample_scene_spec(1, i).band for i in range(0, 24, 6)}
    tiers = {mixture.sample_scene_spec(1, i * 24).tier for i in range(3)}
    assert groups == set(mixture.GROUPS)
    assert bands == {0, 1, 2, 3}
    assert tiers == set(mixture.TIERS)


def test_scene_spec_json_roundtrip():
    spec = _spec(3)
    again = mixture.SceneSpec.from_json(spec.to_json())

    assert again.to_json() == spec.to_json()


def test_render_is_deterministic():
    a = mixture.render_scene(_spec(1))
    b = mixture.render_scene(_spec(1))

    np.testing.assert_array_equal(a.mic, b.mic)
    np.testing.assert_array_equal(a.clean, b.clean)


def test_noise_level_is_calibrated_to_spec():
    spec = _spec(2)  # tank, band 0
    scene = mixture.render_scene(spec)

    assert scene.meta["noise_spl_db"] == pytest.approx(spec.noise_spl_db, abs=0.1)
    assert scene.mic.shape == (2, 2 * SR)
    assert scene.clean.shape == (2 * SR,)
    assert np.isfinite(scene.mic).all()


def test_clean_target_is_speech_only_and_calibrated():
    spec = _spec(0)
    scene = mixture.render_scene(spec)
    # Active-speech rms in dBFS: SPL - (AOP + 3 dB sine crest) up to filter loss.
    rms = speech_synth.active_speech_rms(scene.clean.astype(np.float64), SR)
    expected_dbfs = spec.speech_spl_db - 120.0 - 3.0
    assert 20 * np.log10(rms) == pytest.approx(expected_dbfs, abs=2.5)


def test_high_spl_scene_clips_and_low_spl_does_not():
    low = mixture.render_scene(_spec(0))  # band 0
    high = mixture.render_scene(_spec(18))  # band 3

    assert low.meta["clip_fraction_front"] < 0.05
    assert high.meta["clip_fraction_front"] > 0.3
    assert high.meta["snr_db"] < low.meta["snr_db"]


def test_real_speech_tier_requires_pool():
    with pytest.raises(ValueError, match="SpeechPool"):
        mixture.render_scene(_spec(0, tier="S2"))


def test_scene_io_roundtrip(tmp_path):
    scene = mixture.render_scene(_spec(4))
    hashes = mixture.write_scene(scene, tmp_path)
    back = mixture.read_scene(tmp_path / scene.spec.scene_id)

    assert set(hashes) == {"mic.flac", "clean.flac"}
    np.testing.assert_array_equal(back.mic, scene.mic)  # 24-bit grid is lossless
    np.testing.assert_allclose(back.clean, scene.clean, atol=2e-7)  # 24-bit
    assert back.spec.to_json() == scene.spec.to_json()
