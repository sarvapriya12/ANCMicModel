import pytest

from highspl.registry import (
    MODEL_REGISTRY,
    create_model,
    register_model,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Keep registry tests isolated from one another."""
    MODEL_REGISTRY.clear()
    yield
    MODEL_REGISTRY.clear()


def test_register_model_stores_factory():
    def factory(value=10):
        return {"value": value}

    decorator = register_model("dummy")
    registered = decorator(factory)

    assert registered is factory
    assert "dummy" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dummy"] is factory


def test_create_model_calls_registered_factory():
    @register_model("dummy")
    def factory(value=10):
        return {"value": value}

    result = create_model("dummy", value=42)

    assert result == {"value": 42}


def test_create_model_passes_kwargs():
    @register_model("dummy")
    def factory(name, count):
        return {
            "name": name,
            "count": count,
        }

    result = create_model(
        "dummy",
        name="test",
        count=3,
    )

    assert result["name"] == "test"
    assert result["count"] == 3


def test_duplicate_registration_is_rejected():
    @register_model("dummy")
    def first():
        return 1

    with pytest.raises(KeyError, match="already registered"):

        @register_model("dummy")
        def second():
            return 2

    assert MODEL_REGISTRY["dummy"] is first


def test_unknown_model_raises_useful_error():
    @register_model("available")
    def factory():
        return object()

    with pytest.raises(ValueError, match="Unknown model 'missing'"):
        create_model("missing")

    with pytest.raises(ValueError, match="available"):
        create_model("missing")


def test_empty_model_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        register_model("")


def test_whitespace_model_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        register_model("   ")


def test_model_name_is_normalized():
    @register_model("  dummy  ")
    def factory():
        return 123

    assert create_model("dummy") == 123
    assert create_model("  dummy  ") == 123
