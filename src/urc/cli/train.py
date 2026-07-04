from __future__ import annotations

from pathlib import Path

import typer

from urc.algorithms import algorithms as algorithm_registry
from urc.bridges import bridges as bridge_registry
from urc.config import ConfigError, overrides_to_dict, resolve_config
from urc.core.environments import resolve_environment
from urc.core.plugins import load_all_plugins

_PROJECT_OPTION = typer.Option(
    Path("urc.yaml"), "--project", help="Ruta al YAML de configuración del proyecto."
)
_EXPERIMENT_OPTION = typer.Option(
    None,
    "--experiment",
    "-e",
    help="Ruta a un YAML de experimento a aplicar sobre la config del proyecto.",
)
_SET_OPTION = typer.Option(
    [],
    "--set",
    help="Override puntual clave=valor (p. ej. hyperparameters.learning_rate=1e-4). Repetible.",
)


def train(
    project: Path = _PROJECT_OPTION,
    experiment: Path | None = _EXPERIMENT_OPTION,
    set_: list[str] = _SET_OPTION,
    resume: Path | None = typer.Option(
        None, "--resume", help="Ruta a un checkpoint desde el que reanudar el entrenamiento."
    ),
) -> None:
    """Entrena una política uniendo el bridge y el algoritmo configurados."""
    try:
        config = resolve_config(
            project_path=project,
            experiment_path=experiment,
            overrides=overrides_to_dict(set_),
        )
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    load_all_plugins()

    try:
        bridge_cls = bridge_registry.get(config.bridge)
    except (KeyError, ImportError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    try:
        algorithm_cls = algorithm_registry.get(config.algo)
    except (KeyError, ImportError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    environments_config = {name: entry.model_dump() for name, entry in config.environments.items()}
    env_spec = resolve_environment(config.env, environments_config)
    checkpoint_dir = Path(config.output_dir) / env_spec.name

    config_dict = config.model_dump()
    config_dict["checkpoint_dir"] = str(checkpoint_dir)
    config_dict["resume_from"] = str(resume) if resume else None

    typer.echo(f"Bridge: {config.bridge}  Algoritmo: {config.algo}  Entorno: {env_spec.name}")
    typer.echo(f"Checkpoints en: {checkpoint_dir}")

    bridge_options = {**config.bridge_options, **env_spec.bridge_options}
    bridge = bridge_cls(**bridge_options)
    algorithm = algorithm_cls()
    try:
        algorithm.train(bridge, env_spec, config_dict)
    finally:
        bridge.close()

    typer.echo("Entrenamiento terminado.")
