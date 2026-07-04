from __future__ import annotations

from typing import Generic, TypeVar

from urc.core.contracts import AlgorithmBackend, BridgeAdapter, EnvironmentSpec

T = TypeVar("T")


class Registry(Generic[T]):
    """Registro de componentes intercambiables (bridges, algoritmos, entornos...) por nombre."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T | None = None):
        if value is not None:
            self._add(name, value)
            return value

        def decorator(value_: T) -> T:
            self._add(name, value_)
            return value_

        return decorator

    def _add(self, name: str, value: T) -> None:
        if name in self._items:
            raise ValueError(f"Ya hay un/a {self._kind} registrado/a con el nombre '{name}'")
        self._items[name] = value

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            disponibles = ", ".join(self.names()) or "(ninguno)"
            raise KeyError(f"No existe {self._kind} '{name}'. Disponibles: {disponibles}") from None

    def create(self, name: str, /, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)


bridges: Registry[type[BridgeAdapter]] = Registry("bridge")
algorithms: Registry[type[AlgorithmBackend]] = Registry("algoritmo")
environments: Registry[EnvironmentSpec] = Registry("entorno")
