from __future__ import annotations

from pathlib import Path

import typer

from urc.algorithms import algorithms as algorithm_registry
from urc.bridges import bridges as bridge_registry
from urc.cli._shared import EXPERIMENT_OPTION, PROJECT_OPTION, SET_OPTION, get_or_exit
from urc.config import ConfigError, overrides_to_dict, resolve_config
from urc.core.environments import resolve_environment
from urc.core.plugins import load_all_plugins
from urc.core.runs import write_run_info


def train(
    project: Path = PROJECT_OPTION,
    experiment: Path | None = EXPERIMENT_OPTION,
    set_: list[str] = SET_OPTION,
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

    bridge_cls = get_or_exit(bridge_registry, config.bridge)
    algorithm_cls = get_or_exit(algorithm_registry, config.algo)

    environments_config = {name: entry.model_dump() for name, entry in config.environments.items()}
    env_spec = resolve_environment(config.env, environments_config)
    checkpoint_dir = Path(config.output_dir) / env_spec.name

    config_dict = config.model_dump()
    config_dict["checkpoint_dir"] = str(checkpoint_dir)
    config_dict["resume_from"] = str(resume) if resume else None

    typer.echo(f"Bridge: {config.bridge}  Algoritmo: {config.algo}  Entorno: {env_spec.name}")
    typer.echo(f"Checkpoints en: {checkpoint_dir}")

    bridge_options = {**config.bridge_options, **env_spec.bridge_options}
    write_run_info(
        checkpoint_dir,
        bridge=config.bridge,
        bridge_options=bridge_options,
        algo=config.algo,
        env=env_spec.name,
    )

    bridge = bridge_cls(**bridge_options)
    algorithm = algorithm_cls()
    try:
        algorithm.train(bridge, env_spec, config_dict)
    except ImportError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    finally:
        bridge.close()

    typer.echo("Entrenamiento terminado.")
