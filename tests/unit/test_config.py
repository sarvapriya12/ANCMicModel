import os
from pathlib import Path

import pytest

from highspl.config import (
    load_config,
    project_root,
    resolve_path,
    validate_config,
)


def test_load_base_config():
    config = load_config()

    assert config.project.name == "high-spl-se"
    assert config.signal.capture_rate_hz == 48000
    assert config.model.variant == "fastenhancer_b"
    assert config.model.internal_rate_hz == 48000
    assert config.pipeline.frontend.mode == "robust"


def test_base_config_is_valid():
    config = load_config()

    validate_config(config)


def test_load_config_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.project.name == "high-spl-se"


def test_overrides_are_applied_and_validated():
    config = load_config(overrides=["pipeline.frontend.mode=gsc"])

    assert config.pipeline.frontend.mode == "gsc"
    validate_config(config)


def test_unknown_key_is_rejected():
    with pytest.raises(Exception, match="not_a_key"):
        load_config(overrides=["pipeline.not_a_key=1"])


def test_project_root_has_pyproject():
    assert (project_root() / "pyproject.toml").is_file()


def test_resolve_path_relative_and_absolute(tmp_path):
    assert resolve_path("configs/base.yaml").is_file()
    absolute = tmp_path / "x.wav"
    assert resolve_path(absolute) == absolute


def test_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HIGHSPL_ROOT", str(tmp_path))

    assert project_root() == Path(os.environ["HIGHSPL_ROOT"]).resolve()


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


def test_invalid_frontend_mode():
    config = load_config()
    config.pipeline.frontend.mode = "magic"

    with pytest.raises(ValueError, match="frontend.mode"):
        validate_config(config)


def test_block_size_must_match_hop():
    config = load_config()
    config.runtime.block_size = 500

    with pytest.raises(ValueError, match="multiple of model.hop_size"):
        validate_config(config)
