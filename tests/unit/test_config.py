import pytest

from highspl.config import load_config, validate_config


def test_load_base_config():
    config = load_config()

    assert config.project.name == "high-spl-se"
    assert config.signal.capture_rate_hz == 32000
    assert config.model.variant == "dpdfnet2_48khz_hr"
    assert config.model.internal_rate_hz == 48000


def test_base_config_is_valid():
    config = load_config()

    validate_config(config)


def test_invalid_target_band():
    config = load_config()
    config.signal.target_band_hz = config.signal.capture_rate_hz

    with pytest.raises(ValueError, match="Nyquist"):
        validate_config(config)


def test_invalid_nlms_step_size():
    config = load_config()
    config.nlms.step_size = 0

    with pytest.raises(ValueError, match="step_size"):
        validate_config(config)


def test_invalid_learning_rate():
    config = load_config()
    config.training.learning_rate = 0

    with pytest.raises(ValueError, match="learning_rate"):
        validate_config(config)
