from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# --- Config Dataclasses ---


@dataclass
class ProjectConfig:
    name: str = "high-spl-se"
    seed: int = 1337
    version: str = "0.1.0"


@dataclass
class SignalConfig:
    capture_rate_hz: int = 32000
    target_band_hz: int = 15000
    channels_in: int = 2
    channels_out: int = 1
    dtype: str = "float32"


@dataclass
class NLMSConfig:
    enabled: bool = True
    filter_length: int = 256
    step_size: float = 0.25
    leakage: float = 1.0e-5
    freeze_ratio_db: float = 12.0
    min_reference_power: float = 1.0e-8


@dataclass
class ModelConfig:
    family: str = "dpdfnet"
    variant: str = "dpdfnet2_48khz_hr"
    checkpoint: str = "checkpoints/manifested/model.pth"
    attn_limit_db: float = 12.0
    internal_rate_hz: int = 48000


@dataclass
class RuntimeConfig:
    chunk_ms: int = 10
    num_threads: int = 4
    device: str = "cpu"
    backend: str = "pytorch"


@dataclass
class TrainingConfig:
    segment_seconds: float = 4.0
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    mixed_precision: bool = True


@dataclass
class MetricsConfig:
    use_pesq: bool = True
    use_stoi: bool = True
    use_si_sdr: bool = True
    use_dnsmos: bool = True
    use_wer: bool = True


@dataclass
class AppConfig:
    project: ProjectConfig
    signal: SignalConfig
    nlms: NLMSConfig
    model: ModelConfig
    runtime: RuntimeConfig
    training: TrainingConfig
    metrics: MetricsConfig


# --- Core Functions ---


def load_config(path: str | Path = "configs/base.yaml") -> DictConfig:
    """Load and validate the application configuration."""
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    schema = OmegaConf.structured(AppConfig)
    loaded = OmegaConf.load(config_path)
    config = OmegaConf.merge(schema, loaded)
    OmegaConf.set_struct(config, True)

    return config


def validate_config(config: DictConfig) -> None:
    """Master validation function delegating to component validators."""
    _validate_signal(config.signal)
    _validate_nlms(config.nlms)
    _validate_model(config.model)
    _validate_runtime(config.runtime)
    _validate_training(config.training)


# --- Modular Validation Helpers ---


def _validate_signal(signal: DictConfig) -> None:
    if signal.capture_rate_hz <= 0:
        raise ValueError("signal.capture_rate_hz must be positive")
    if signal.target_band_hz <= 0:
        raise ValueError("signal.target_band_hz must be positive")
    if signal.target_band_hz >= signal.capture_rate_hz / 2:
        raise ValueError("signal.target_band_hz must be below the Nyquist frequency")
    if signal.channels_in < 1:
        raise ValueError("signal.channels_in must be >= 1")
    if signal.channels_out < 1:
        raise ValueError("signal.channels_out must be >= 1")


def _validate_nlms(nlms: DictConfig) -> None:
    if nlms.filter_length <= 0:
        raise ValueError("nlms.filter_length must be positive")
    if not (0.0 < nlms.step_size <= 1.0):
        raise ValueError("nlms.step_size must be in (0, 1]")


def _validate_model(model: DictConfig) -> None:
    if model.internal_rate_hz <= 0:
        raise ValueError("model.internal_rate_hz must be positive")


def _validate_runtime(runtime: DictConfig) -> None:
    if runtime.chunk_ms <= 0:
        raise ValueError("runtime.chunk_ms must be positive")


def _validate_training(training: DictConfig) -> None:
    if training.segment_seconds <= 0:
        raise ValueError("training.segment_seconds must be positive")
    if training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if training.epochs <= 0:
        raise ValueError("training.epochs must be positive")
    if training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay must be >= 0")
    if training.grad_clip_norm <= 0:
        raise ValueError("training.grad_clip_norm must be positive")
