from __future__ import annotations

from typing import Any

from urc.core.contracts import EnvironmentSpec
from urc.core.registry import environments


def register_environments_from_config(environments_config: dict[str, Any]) -> None:
    """Registra como `EnvironmentSpec` cada entorno declarado en `urc.yaml`.

    Usa `Registry.set` (upsert) a propósito, no `register`: la config es la
    fuente de verdad y se puede resolver más de una vez en el mismo proceso
    (p. ej. `urc train` seguido de `urc env list` en la misma sesión de
    tests), así que volver a "registrar" un entorno con los mismos datos debe
    ser normal, no un error.
    """
    for name, entry in environments_config.items():
        spec = EnvironmentSpec(
            name=name,
            build_path=entry.get("build_path"),
            bridge_options=entry.get("bridge_options", {}),
            parameters=entry.get("parameters", {}),
            curriculum=entry.get("curriculum") or None,
        )
        environments.set(name, spec)


def resolve_environment(name: str | None, environments_config: dict[str, Any]) -> EnvironmentSpec:
    """Registra los entornos declarados y devuelve el elegido por `name`.

    Si `name` no está entre los entornos declarados (o no se declaró
    ninguno), se usa como una etiqueta suelta sin opciones extra — así los
    proyectos que ya usaban `env: mi-experimento` solo como nombre de carpeta
    (antes de que existiera esta sección) siguen funcionando igual.
    """
    register_environments_from_config(environments_config)
    if name and name in environments.names():
        return environments.get(name)
    return EnvironmentSpec(name=name or "default")
