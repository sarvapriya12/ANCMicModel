"""Stage-1 DSP front-ends behind a single stateful interface.

Every front-end maps a (primary, reference) pair of equal-length float32
blocks to one mono block of the same length, owning its own state.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .fdaf import BattlefieldFDAF
from .nlms import NLMS, VSSNLMS, RobustNLMS


class FrontEnd(Protocol):
    name: str

    def reset(self) -> None: ...

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray: ...

    def adaptation_frozen(self) -> bool:
        """True if the last block did not adapt (diagnostics only)."""
        ...


class BypassFrontEnd:
    name = "bypass"

    def reset(self) -> None:
        return None

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.asarray(primary, dtype=np.float32).copy()

    def adaptation_frozen(self) -> bool:
        return False


class _FilterFrontEnd:
    """Wraps a legacy filter exposing reset() -> state / process(d, x, state)."""

    def __init__(self, name: str, filt: Any) -> None:
        self.name = name
        self._filter = filt
        self._state = filt.reset()

    def reset(self) -> None:
        self._state = self._filter.reset()

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        out, self._state = self._filter.process(primary, reference, self._state)
        return out

    def adaptation_frozen(self) -> bool:
        return int(getattr(self._state, "freeze_remaining", 0)) > 0


def make_frontend(mode: str, **kwargs: Any) -> FrontEnd:
    """Build a front-end by mode name.

    ``mode`` is one of bypass | robust | vss | nlms | fdaf, plus the
    beamformer modes (diff | adiff | gsc) registered by ``dsp.beamformer``.
    Extra keyword arguments are forwarded to the underlying implementation.
    """
    mode = mode.lower()
    if mode == "bypass":
        return BypassFrontEnd()
    if mode == "robust":
        return _FilterFrontEnd("robust", RobustNLMS(**kwargs))
    if mode == "vss":
        return _FilterFrontEnd("vss", VSSNLMS(**kwargs))
    if mode == "nlms":
        return _FilterFrontEnd("nlms", NLMS(**kwargs))
    if mode == "fdaf":
        kwargs.setdefault("block_size", 128)
        kwargs.setdefault("num_partitions", 4)
        kwargs.setdefault("step_size", 0.05)
        return _FilterFrontEnd("fdaf", BattlefieldFDAF(**kwargs))

    if mode in ("diff", "adiff", "gsc"):
        from . import beamformer  # noqa: F401  (registers the factories)

    factory = _BEAMFORMER_FACTORIES.get(mode)
    if factory is not None:
        return factory(**kwargs)

    raise ValueError(f"Unknown front-end mode: {mode!r}")


# Filled in by highspl.dsp.beamformer on import (avoids a circular import).
_BEAMFORMER_FACTORIES: dict[str, Any] = {}


def register_frontend(mode: str, factory: Any) -> None:
    _BEAMFORMER_FACTORIES[mode] = factory
