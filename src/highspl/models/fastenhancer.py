import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import ModelState, StreamingEnhancer


@dataclass
class FastEnhancerConfig:
    """Configuration for the FastEnhancer streaming adapter."""

    model_kwargs: dict[str, Any]
    checkpoint_path: str | None = None
    onnx_path: str | None = None
    backend: str = "pytorch"
    device: str = "cpu"
    upstream_root: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"pytorch", "onnxruntime", "tensorrt"}:
            raise ValueError("backend must be one of: pytorch, onnxruntime, tensorrt")

        if not self.model_kwargs:
            raise TypeError("model_kwargs must not be empty")

        sample_rate = self.model_kwargs.get("sample_rate", 48_000)
        if sample_rate != 48_000:
            raise ValueError("FastEnhancer adapter requires 48 kHz")

        hop_size = self.model_kwargs.get("hop_size")
        if hop_size is None or hop_size <= 0:
            raise ValueError("model_kwargs must contain a positive hop_size")

        if (
            self.checkpoint_path is not None
            and not Path(self.checkpoint_path).is_file()
        ):
            raise FileNotFoundError(
                f"FastEnhancer checkpoint not found: {self.checkpoint_path}"
            )

        if self.onnx_path is not None and not Path(self.onnx_path).is_file():
            raise FileNotFoundError(
                f"FastEnhancer ONNX model not found: {self.onnx_path}"
            )


class FastEnhancerAdapter(StreamingEnhancer):
    """Streaming adapter around the official FastEnhancer implementation.

    The adapter owns:
    - input buffering
    - STFT state
    - iSTFT state
    - FastEnhancer RNNFormer cache state

    The application pipeline only sees the generic StreamingEnhancer API.
    """

    sample_rate_in = 48_000
    sample_rate_out = 48_000

    def __init__(self, config: FastEnhancerConfig) -> None:
        self.config = config
        self.hop_size = int(config.model_kwargs["hop_size"])
        self.device = config.device

        self._model: Any = None
        self._cache: list[Any] = []
        self._input_buffer = np.empty(0, dtype=np.float32)

        if config.backend == "tensorrt":
            raise NotImplementedError(
                "TensorRT backend is not officially provided by FastEnhancer"
            )

        if config.backend == "pytorch" and config.checkpoint_path is not None:
            self._load_pytorch_model()

        if config.backend == "onnxruntime" and config.onnx_path is not None:
            self._load_onnx_model()

    def _resolve_upstream_root(self) -> Path:
        """Resolve the local official FastEnhancer repository."""
        if self.config.upstream_root is not None:
            root = Path(self.config.upstream_root).resolve()
        else:
            # Expected local layout:
            # D:\SIH\
            # ├── ANCMicModel
            # └── fastenhancer
            project_root = Path(__file__).resolve().parents[3]
            root = project_root.parent / "fastenhancer"

        if not root.is_dir():
            raise FileNotFoundError(
                f"FastEnhancer upstream repository not found: {root}"
            )

        return root

    def _import_upstream_onnx_model(self) -> Any:
        """Import the official FastEnhancer ONNXModel class."""
        root = self._resolve_upstream_root()

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        module = importlib.import_module("models.fastenhancer.default.model")
        return module.ONNXModel

    def _load_pytorch_model(self) -> None:
        """Load the official streaming-capable FastEnhancer model."""
        import torch

        if self.config.checkpoint_path is None:
            raise TypeError("checkpoint_path is required for the PyTorch backend")

        model_class = self._import_upstream_onnx_model()

        model_kwargs = self.config.model_kwargs.copy()
        model_kwargs.pop("sample_rate", None)

        model = model_class(**model_kwargs)

        checkpoint = torch.load(
            self.config.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError("FastEnhancer checkpoint must contain a dictionary")

        if "model" not in checkpoint:
            raise KeyError("FastEnhancer checkpoint does not contain a 'model' entry")

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        model.remove_weight_reparameterizations()
        model.eval()
        model.to(self.device)
        model.flatten_parameters()

        self._model = model
        self._cache = []

    def _load_onnx_model(self) -> None:
        """Load an ONNX Runtime model.

        Full streaming-cache initialization is implemented after the
        PyTorch backend has been validated.
        """
        import onnxruntime as ort

        if self.config.onnx_path is None:
            raise TypeError("onnx_path is required for the ONNX Runtime backend")

        providers = ["CPUExecutionProvider"]

        if self.device.startswith("cuda"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ]

        self._model = ort.InferenceSession(
            self.config.onnx_path,
            providers=providers,
        )
        self._cache = []

    def reset(self) -> ModelState:
        """Reset input buffering and all model streaming state."""
        self._input_buffer = np.empty(0, dtype=np.float32)

        if self.config.backend == "pytorch":
            if self._model is None:
                self._cache = []
            else:
                import torch

                x = torch.zeros(
                    1,
                    1,
                    device=self.device,
                    dtype=torch.float32,
                )

                cache_stft, cache_istft = self._model.stft.initialize_cache(x)

                cache_model = self._model.initialize_cache(x)

                self._cache = [
                    cache_stft,
                    cache_istft,
                    *cache_model,
                ]

        elif self.config.backend == "onnxruntime":
            if self._model is None:
                self._cache = []
            else:
                self._cache = []
                # First input is wav_in, remaining are caches
                for inp in self._model.get_inputs()[1:]:
                    shape = inp.shape
                    self._cache.append(np.zeros(shape, dtype=np.float32))

        return ModelState(backend=self._cache)

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        state: ModelState,
    ) -> tuple[np.ndarray, ModelState]:
        """Buffer arbitrary input chunks and process complete model hops."""

        if sample_rate != self.sample_rate_in:
            raise ValueError(
                f"FastEnhancer expects {self.sample_rate_in} Hz, got {sample_rate} Hz"
            )

        if not isinstance(audio, np.ndarray):
            raise TypeError("audio must be a numpy.ndarray")

        if audio.ndim != 1:
            raise ValueError("FastEnhancer currently expects mono 1-D audio")

        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite values")

        audio = np.asarray(audio, dtype=np.float32)

        if state.backend is not None:
            self._cache = state.backend

        if audio.size:
            self._input_buffer = np.concatenate([self._input_buffer, audio])

        outputs: list[np.ndarray] = []

        while self._input_buffer.size >= self.hop_size:
            chunk = self._input_buffer[: self.hop_size]
            self._input_buffer = self._input_buffer[self.hop_size :]

            if self._model is None:
                raise RuntimeError(
                    "FastEnhancer backend not initialized. "
                    "Provide checkpoint_path for the PyTorch backend."
                )

            output = self._process_hop(chunk)
            outputs.append(np.asarray(output, dtype=np.float32).reshape(-1))

        if outputs:
            enhanced = np.concatenate(outputs)
        else:
            enhanced = np.empty(0, dtype=np.float32)

        state.backend = self._cache

        return enhanced, state

    def flush(
        self,
        state: ModelState,
    ) -> tuple[np.ndarray, ModelState]:
        """Process the final partial hop with zero padding."""
        remaining = int(self._input_buffer.size)

        if remaining == 0:
            return np.empty(0, dtype=np.float32), state

        padded = np.zeros(self.hop_size, dtype=np.float32)
        padded[:remaining] = self._input_buffer

        self._input_buffer = np.empty(0, dtype=np.float32)

        if self._model is None:
            raise RuntimeError("FastEnhancer backend not initialized")

        output = self._process_hop(padded)

        state.backend = self._cache

        output = np.asarray(
            output,
            dtype=np.float32,
        ).reshape(-1)

        return output[:remaining], state

    def _process_hop(self, chunk: np.ndarray) -> np.ndarray:
        """Process one exact FastEnhancer hop."""
        if self.config.backend == "onnxruntime":
            return self._process_onnx(chunk)

        return self._process_pytorch(chunk)

    def _process_pytorch(self, chunk: np.ndarray) -> np.ndarray:
        """Run one hop through the official streaming PyTorch path."""
        import torch

        if len(self._cache) != 5:
            raise RuntimeError(
                "FastEnhancer streaming cache is not initialized. "
                "Call reset() before process()."
            )

        cache_stft = self._cache[0]
        cache_istft = self._cache[1]
        cache_model = self._cache[2:]

        tensor = torch.from_numpy(chunk.reshape(1, self.hop_size)).to(
            self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            spec_in, cache_stft = self._model.stft(
                tensor,
                cache_stft,
            )

            spec_out, *cache_model = self._model(
                spec_in,
                *cache_model,
            )

            wav_out, cache_istft = self._model.stft.inverse(
                spec_out,
                cache_istft,
            )

        self._cache = [
            cache_stft,
            cache_istft,
            *cache_model,
        ]

        return wav_out.detach().cpu().numpy().reshape(-1)

    def _process_onnx(self, chunk: np.ndarray) -> np.ndarray:
        """Run one ONNX Runtime streaming hop."""
        if not hasattr(self._model, "run"):
            raise RuntimeError("ONNX Runtime backend is not initialized")

        inputs = {
            "wav_in": chunk.reshape(1, self.hop_size),
        }

        for index, cache in enumerate(self._cache):
            inputs[f"cache_in_{index}"] = cache

        outputs = self._model.run(
            None,
            inputs,
        )

        wav_out = outputs[0]
        self._cache = list(outputs[1:])

        return np.asarray(
            wav_out,
            dtype=np.float32,
        ).reshape(-1)
