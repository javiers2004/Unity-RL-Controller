from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from urc.config.schema import UrcConfig


class ConfigError(RuntimeError):
    """La configuración no se pudo cargar o no es válida; el mensaje ya es legible."""


def _load_defaults() -> dict[str, Any]:
    text = importlib.resources.files("urc.config").joinpath("defaults.yaml").read_text(
        encoding="utf-8"
    )
    return yaml.safe_load(text) or {}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"No se pudo leer '{path}': {error}") from error

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"'{path}': el YAML raíz debe ser un mapeo (clave: valor), no {type(data).__name__}"
        )
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Combina `override` sobre `base`; los dicts anidados se fusionan, no se sustituyen."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(raw: str) -> Any:
    """Interpreta un valor de override como int/float/bool/None, o YAML si no encaja.

    No se delega directamente en `yaml.safe_load`: la especificación YAML 1.1 no
    reconoce '3e-4' como float (exige un punto decimal, tipo '3.0e-4'), y esa es
    justo la forma más natural de escribir un learning rate en la terminal.
    """
    stripped = raw.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~", ""):
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return yaml.safe_load(raw)


def parse_override(raw: str) -> tuple[str, Any]:
    """Convierte 'clave.anidada=valor' en (ruta, valor python)."""
    if "=" not in raw:
        raise ConfigError(f"Override inválido '{raw}': se espera el formato clave=valor")
    key, _, raw_value = raw.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"Override inválido '{raw}': falta la clave")
    return key, _parse_scalar(raw_value)


def overrides_to_dict(raw_overrides: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in raw_overrides:
        dotted_key, value = parse_override(raw)
        node = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def _format_validation_error(error: ValidationError) -> str:
    lines = ["La configuración no es válida:"]
    for err in error.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(raíz)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def resolve_config(
    *,
    project_path: Path | None = None,
    experiment_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> UrcConfig:
    """Resuelve la config final: defaults de la librería -> proyecto -> experimento -> overrides."""
    merged = _load_defaults()

    if project_path is not None and project_path.exists():
        merged = deep_merge(merged, load_yaml(project_path))

    if experiment_path is not None:
        merged = deep_merge(merged, load_yaml(experiment_path))

    if overrides:
        merged = deep_merge(merged, overrides)

    try:
        return UrcConfig.model_validate(merged)
    except ValidationError as error:
        raise ConfigError(_format_validation_error(error)) from error


def diff_configs(a: UrcConfig, b: UrcConfig) -> dict[str, tuple[Any, Any]]:
    return _diff_dicts(a.model_dump(), b.model_dump())


def _diff_dicts(
    a: dict[str, Any], b: dict[str, Any], prefix: str = ""
) -> dict[str, tuple[Any, Any]]:
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(a) | set(b)):
        path = f"{prefix}.{key}" if prefix else key
        value_a, value_b = a.get(key), b.get(key)
        if isinstance(value_a, dict) and isinstance(value_b, dict):
            differences.update(_diff_dicts(value_a, value_b, path))
        elif value_a != value_b:
            differences[path] = (value_a, value_b)
    return differences
