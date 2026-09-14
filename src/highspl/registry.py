from collections.abc import Callable
from typing import Any

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a model factory under a unique name."""

    if not name or not name.strip():
        raise ValueError("model name must not be empty")

    normalized_name = name.strip()

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if normalized_name in MODEL_REGISTRY:
            raise KeyError(f"model already registered: {normalized_name}")

        MODEL_REGISTRY[normalized_name] = factory
        return factory

    return decorator


def create_model(name: str, /, **kwargs: Any) -> Any:
    """Create a registered model using the supplied keyword arguments."""

    normalized_name = name.strip()

    try:
        factory = MODEL_REGISTRY[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{normalized_name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from exc

    return factory(**kwargs)
