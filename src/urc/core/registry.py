from __future__ import annotations

import importlib
from typing import Generic, TypeVar

from urc.core.contracts import AlgorithmBackend, BridgeAdapter, EnvironmentSpec

T = TypeVar("T")


class Registry(Generic[T]):
    """Registro de componentes intercambiables (bridges, algoritmos, entornos...) por nombre."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}
        self._lazy_modules: dict[str, tuple[str, str]] = {}

    def register(self, name: str, value: T | None = None):
        if value is not None:
            self._add(name, value)
            return value

        def decorator(value_: T) -> T:
            self._add(name, value_)
            return value_

        return decorator

    def register_lazy(self, name: str, module: str, *, install_hint: str) -> None:
        """Declara que `name` existe pero su módulo (con dependencias opcionales
        pesadas) solo se importa la primera vez que se pide de verdad con `get`.
        `install_hint` es lo que se sugiere al usuario si esa importación falla."""
        self._lazy_modules[name] = (module, install_hint)

    def _add(self, name: str, value: T) -> None:
        if name in self._items:
            raise ValueError(f"Ya hay un/a {self._kind} registrado/a con el nombre '{name}'")
        self._items[name] = value

    def set(self, name: str, value: T) -> None:
        """Como `register`, pero sin fallar si `name` ya existe (upsert).

        Pensado para cosas re-derivables de config (p. ej. entornos): a
        diferencia de un plugin de código, que solo debería registrarse una
        vez, un `EnvironmentSpec` se reconstruye cada vez que se resuelve la
        config, y volver a "registrarlo" con los mismos datos es normal, no
        un error de programación.
        """
        self._items[name] = value

    def get(self, name: str) -> T:
        if name not in self._items and name in self._lazy_modules:
            module_name, install_hint = self._lazy_modules[name]
            try:
                importlib.import_module(module_name)
            except ImportError as error:
                raise ImportError(
                    f"{self._kind} '{name}' necesita una dependencia opcional no instalada. "
                    f"Instálala con: {install_hint}"
                ) from error

        try:
            return self._items[name]
        except KeyError:
            disponibles = ", ".join(self.names()) or "(ninguno)"
            raise KeyError(f"No existe {self._kind} '{name}'. Disponibles: {disponibles}") from None

    def create(self, name: str, /, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(set(self._items) | set(self._lazy_modules))


bridges: Registry[type[BridgeAdapter]] = Registry("bridge")
algorithms: Registry[type[AlgorithmBackend]] = Registry("algoritmo")
environments: Registry[EnvironmentSpec] = Registry("entorno")
