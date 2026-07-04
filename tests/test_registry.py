import pytest

from urc.core.registry import Registry


def test_register_as_decorator_and_get_by_name():
    registry: Registry[object] = Registry("cosa")

    @registry.register("dummy")
    class Dummy:
        pass

    assert registry.get("dummy") is Dummy


def test_register_with_explicit_value():
    registry: Registry[object] = Registry("cosa")

    registry.register("dummy", object)

    assert registry.get("dummy") is object


def test_get_unknown_name_raises_key_error_listing_available_names():
    registry: Registry[object] = Registry("cosa")
    registry.register("a", object)
    registry.register("b", object)

    with pytest.raises(KeyError) as excinfo:
        registry.get("c")

    message = str(excinfo.value)
    assert "a" in message
    assert "b" in message


def test_register_duplicate_name_raises_value_error():
    registry: Registry[object] = Registry("cosa")
    registry.register("dummy", object)

    with pytest.raises(ValueError, match="dummy"):
        registry.register("dummy", object)


def test_create_instantiates_registered_class():
    registry: Registry[object] = Registry("cosa")

    @registry.register("greeter")
    class Greeter:
        def __init__(self, name: str) -> None:
            self.name = name

    greeter = registry.create("greeter", name="Ada")

    assert isinstance(greeter, Greeter)
    assert greeter.name == "Ada"


def test_names_lists_registered_entries_sorted():
    registry: Registry[object] = Registry("cosa")
    registry.register("z", object)
    registry.register("a", object)

    assert registry.names() == ["a", "z"]
