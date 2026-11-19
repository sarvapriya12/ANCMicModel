import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# --- Paths ---


def project_root() -> Path:
    """Locate the repository root (directory holding pyproject.toml).

    ``HIGHSPL_ROOT`` overrides the search; otherwise we walk up from this file,
    which works for editable installs and in-tree execution.
    """
    env = os.environ.get("HIGHSPL_ROOT")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


# --- Config Dataclasses ---


@dataclass
class ProjectConfig:
    name: str = "high-spl-se"
    seed: int = 1337
    version: str = "0.1.0"


@dataclass
class SignalConfig:
    capture_rate_hz: int = 48000
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
    pld_threshold_db: float = 12.0
    min_reference_db: float = -40.0
    max_reference_db: float = -3.0


@dataclass
class ModelConfig:
    family: str = "fastenhancer"
    variant: str = "fastenhancer_b"
    checkpoint: str = "models/fastenhancer/b/00500.pth"
    onnx_path: str = "models/fastenhancer/b/fastenhancer_b_streaming.onnx"
    hop_size: int = 512
    internal_rate_hz: int = 48000


@dataclass
class RuntimeConfig:
    # Samples per processing block. Must equal the model hop (512 = 10.67 ms).
    block_size: int = 512
    num_threads: int = 1
    device: str = "cpu"
    backend: str = "onnxruntime"


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
    use_dnsmos: bool = False
    use_wer: bool = False


@dataclass
class FrontendConfig:
    """Stage-1 DSP. See highspl.dsp.frontend for the available modes."""

    mode: str = "robust"  # bypass|robust|vss|nlms|fdaf|diff|adiff|gsc
    # Geometry of the back-to-back pod (primary faces the lips, reference
    # faces away). Used by the differential / GSC front-ends.
    mic_spacing_m: float = 0.015
    mouth_distance_m: float = 0.055
    # Fade the beamformer in with the noise level (primary mic only when quiet).
    blend: bool = False
    blend_floor_lo_dbfs: float = -34.0
    blend_floor_hi_dbfs: float = -24.0


@dataclass
class LevelConfig:
    # Linear gain applied after the model. -1.94 dB == the legacy x0.8.
    output_gain_db: float = -1.94
    # Streaming AGC in front of the neural network.
    agc_enabled: bool = False
    agc_target_dbfs: float = -24.0
    agc_max_gain_db: float = 24.0
    agc_min_gain_db: float = -24.0
    # Hearing-safety limiter on the output.
    limiter_enabled: bool = False
    limiter_ceiling_dbfs: float = -1.0


@dataclass
class GuardConfig:
    """Clipping / impulse / bad-sample handling in front of the DSP."""

    sanitize: bool = True
    clip_threshold: float = 0.98
    clip_diversity: bool = False
    impulse_duck: bool = False


@dataclass
class CalibrationConfig:
    enabled: bool = False
    update_seconds: float = 3.0


@dataclass
class PipelineConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    level: LevelConfig = field(default_factory=LevelConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    nlms: NLMSConfig = field(default_factory=NLMSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# --- Core Functions ---


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> DictConfig:
    """Load and validate the application configuration.

    ``path`` defaults to ``configs/base.yaml`` under the project root and may
    be relative to it. ``overrides`` are dotlist strings, e.g.
    ``["pipeline.frontend.mode=gsc"]``.
    """
    config_path = resolve_path(path if path is not None else "configs/base.yaml")

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    schema = OmegaConf.structured(AppConfig)
    loaded = OmegaConf.load(config_path)
    config = OmegaConf.merge(schema, loaded)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(config, True)

    return config


def validate_config(config: DictConfig) -> None:
    """Master validation function delegating to component validators."""
    _validate_signal(config.signal)
    _validate_nlms(config.nlms)
    _validate_model(config.model)
    _validate_runtime(config.runtime, config.model)
    _validate_training(config.training)
    _validate_pipeline(config.pipeline)


# --- Modular Validation Helpers ---

FRONTEND_MODES = {"bypass", "robust", "vss", "nlms", "fdaf", "diff", "adiff", "gsc"}


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
    if model.hop_size <= 0:
        raise ValueError("model.hop_size must be positive")


def _validate_runtime(runtime: DictConfig, model: DictConfig) -> None:
    if runtime.block_size <= 0:
        raise ValueError("runtime.block_size must be positive")
    if runtime.block_size % model.hop_size != 0:
        raise ValueError("runtime.block_size must be a multiple of model.hop_size")
    if runtime.num_threads < 0:
        raise ValueError("runtime.num_threads must be >= 0")


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


def _validate_pipeline(pipeline: DictConfig) -> None:
    fe = pipeline.frontend
    if fe.mode not in FRONTEND_MODES:
        raise ValueError(
            f"pipeline.frontend.mode must be one of {sorted(FRONTEND_MODES)}, "
            f"got {fe.mode!r}"
        )
    if fe.mic_spacing_m <= 0:
        raise ValueError("pipeline.frontend.mic_spacing_m must be positive")
    if fe.mouth_distance_m <= 0:
        raise ValueError("pipeline.frontend.mouth_distance_m must be positive")
    if fe.blend and fe.blend_floor_lo_dbfs >= fe.blend_floor_hi_dbfs:
        raise ValueError(
            "pipeline.frontend.blend_floor_lo_dbfs must be < blend_floor_hi_dbfs"
        )
    if pipeline.level.agc_min_gain_db > pipeline.level.agc_max_gain_db:
        raise ValueError("pipeline.level.agc_min_gain_db must be <= agc_max_gain_db")
    if not (0.0 < pipeline.guard.clip_threshold <= 1.0):
        raise ValueError("pipeline.guard.clip_threshold must be in (0, 1]")
