"""Piezas de CLI compartidas entre `eval`, `record` y `train` (opciones y
resolución de bridge/algoritmo/checkpoint comunes a los tres)."""

from __future__ import annotations

from pathlib import Path

import typer

from urc.algorithms import algorithms as algorithm_registry
from urc.bridges import bridges as bridge_registry
from urc.config import ConfigError, overrides_to_dict, resolve_config
from urc.core.contracts import AlgorithmBackend, BridgeAdapter, Policy
from urc.core.plugins import load_all_plugins
from urc.core.runs import load_run_info

PROJECT_OPTION = typer.Option(
    Path("urc.yaml"), "--project", help="Ruta al YAML de configuración del proyecto."
)
EXPERIMENT_OPTION = typer.Option(
    None,
    "--experiment",
    "-e",
    help="Ruta a un YAML de experimento a aplicar sobre la config del proyecto.",
)
SET_OPTION = typer.Option(
    [],
    "--set",
    help="Override puntual clave=valor (p. ej. hyperparameters.learning_rate=1e-4). Repetible.",
)


def load_bridge_and_policy(
    checkpoint: Path,
    *,
    project: Path,
    experiment: Path | None,
    set_: list[str],
) -> tuple[BridgeAdapter, Policy]:
    """Reconstruye el bridge y carga la política de un checkpoint.

    Usa `run_info.json` (si `urc train` lo dejó junto al checkpoint) como
    config de partida, para no tener que repetir `--project`/`--set` con lo
    mismo que se usó para entrenar; `--project`/`--experiment`/`--set`
    explícitos siguen pudiendo sobreescribirlo.
    """
    run_info = load_run_info(checkpoint)
    try:
        config = resolve_config(
            project_path=project,
            experiment_path=experiment,
            overrides=overrides_to_dict(set_),
            extra_defaults=run_info,
        )
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    load_all_plugins()

    bridge_cls = get_or_exit(bridge_registry, config.bridge)
    algorithm_cls: type[AlgorithmBackend] = get_or_exit(algorithm_registry, config.algo)

    bridge = bridge_cls(**config.bridge_options)
    algorithm = algorithm_cls()
    policy = algorithm.load(str(checkpoint))
    return bridge, policy


def get_or_exit(registry, name: str):
    try:
        return registry.get(name)
    except (KeyError, ImportError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
