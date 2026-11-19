"""Lab-facing wrapper around :class:`highspl.pipeline.StreamingPipeline`.

The lab UI keeps its historical ``HighSPLPipeline`` interface (a ``dsp_mode``
attribute and a metrics dict per block); all processing lives in the shared
library pipeline so the UI, the evaluator and the Pi run identical code.
"""

import numpy as np

from highspl.evaluation.runner import default_preset, load_pipeline_config
from highspl.pipeline import StreamingPipeline

_MODE_ALIASES = {"AI": "bypass"}


class HighSPLPipeline:
    def __init__(
        self, preset: str | None = None, config_overrides: list[str] | None = None
    ):
        self.preset = preset or default_preset()
        self._cfg = load_pipeline_config(self.preset, config_overrides)
        self._pipe = StreamingPipeline(self._cfg)

    # -- mode selection (UI thread) ------------------------------------
    @property
    def dsp_mode(self) -> str:
        return self._pipe.mode.upper()

    @dsp_mode.setter
    def dsp_mode(self, mode: str) -> None:
        mode = _MODE_ALIASES.get(mode.upper(), mode).lower()
        if mode != self._pipe.mode:
            # Fresh state for the newly selected front-end.
            self._pipe.set_mode(mode)

    def reset(self) -> None:
        """Reset all streaming state. Call on every stream start."""
        self._pipe.reset()

    # -- streaming -------------------------------------------------------
    def process(self, primary: np.ndarray, reference: np.ndarray):
        eps = 1e-8
        res = self._pipe.process(primary, reference)
        d, x, out = res.primary, res.reference, res.enhanced

        p_energy = float(np.sum(d**2))
        r_energy = float(np.sum(x**2))
        coherence = abs(float(np.sum(d * x))) / (np.sqrt(p_energy * r_energy) + eps)
        out_energy = max(float(np.sum(out**2)), eps)
        nr_db = 10 * np.log10(out_energy / (p_energy + eps))

        return {
            "waveform": primary,
            "waveform_ref": reference,
            "enhanced_audio": out,
            "noise_reduction_db": nr_db,
            "coherence": coherence,
            "latency": {
                "dsp": res.telemetry["time_dsp_ms"],
                "ai": res.telemetry["time_ai_ms"],
            },
            "telemetry": res.telemetry,
        }
