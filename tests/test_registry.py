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


def test_names_includes_lazy_entries_before_they_are_imported():
    registry: Registry[object] = Registry("cosa")
    registry.register("eager", object)
    registry.register_lazy("lazy-one", "some.module", install_hint="pip install x")

    assert registry.names() == ["eager", "lazy-one"]


def test_get_imports_lazy_module_on_demand(monkeypatch: pytest.MonkeyPatch):
    registry: Registry[object] = Registry("cosa")
    registry.register_lazy("thing", "some.module.path", install_hint="pip install thing")
    imported: list[str] = []

    def fake_import_module(name: str) -> None:
        imported.append(name)
        registry.register("thing", object)  # simula el @registry.register del módulo real

    monkeypatch.setattr("urc.core.registry.importlib.import_module", fake_import_module)

    result = registry.get("thing")

    assert result is object
    assert imported == ["some.module.path"]


def test_get_does_not_reimport_once_already_registered(monkeypatch: pytest.MonkeyPatch):
    registry: Registry[object] = Registry("cosa")
    registry.register("thing", object)
    registry.register_lazy("thing", "some.module.path", install_hint="pip install thing")

    def fail_if_called(name: str) -> None:
        raise AssertionError("no debería reimportar algo ya registrado")

    monkeypatch.setattr("urc.core.registry.importlib.import_module", fail_if_called)

    assert registry.get("thing") is object


def test_get_raises_import_error_with_install_hint_when_lazy_import_fails():
    registry: Registry[object] = Registry("cosa")
    registry.register_lazy(
        "thing", "does.not.exist.module", install_hint='pip install "urc[thing]"'
    )

    with pytest.raises(ImportError, match=r'pip install "urc\[thing\]"'):
        registry.get("thing")
