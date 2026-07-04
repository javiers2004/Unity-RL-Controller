from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from urc.config import ConfigError, resolve_config
from urc.config.loader import load_yaml
from urc.core.environments import register_environments_from_config
from urc.core.registry import environments as environment_registry

env_app = typer.Typer(
    help="Gestiona los entornos/mapas: declararlos, inspeccionarlos y lanzar builds de Unity."
)

_PROJECT_OPTION = typer.Option(
    Path("urc.yaml"), "--project", help="Ruta al YAML de configuración del proyecto."
)


def _load_registered_environments(project: Path) -> None:
    try:
        config = resolve_config(project_path=project)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    register_environments_from_config(
        {name: entry.model_dump() for name, entry in config.environments.items()}
    )


@env_app.command("list")
def list_environments(project: Path = _PROJECT_OPTION) -> None:
    """Lista los entornos declarados en `environments:` de la config del proyecto."""
    _load_registered_environments(project)
    for name in environment_registry.names():
        typer.echo(name)


@env_app.command("describe")
def describe(
    name: str = typer.Argument(..., help="Nombre del entorno, tal como aparece en 'urc env list'."),
    project: Path = _PROJECT_OPTION,
) -> None:
    """Muestra los detalles de un entorno registrado."""
    _load_registered_environments(project)
    try:
        spec = environment_registry.get(name)
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"name: {spec.name}")
    typer.echo(f"build_path: {spec.build_path}")
    typer.echo(f"bridge_options: {spec.bridge_options}")
    typer.echo(f"parameters: {spec.parameters}")
    typer.echo(f"curriculum: {spec.curriculum}")


@env_app.command("create")
def create(
    name: str = typer.Argument(..., help="Nombre del nuevo entorno."),
    build_path: Path | None = typer.Option(
        None, "--build-path", help="Ruta al build de Unity (opcional)."
    ),
    project: Path = _PROJECT_OPTION,
) -> None:
    """Añade un entorno nuevo a la sección `environments:` de la config del proyecto."""
    data: dict[str, Any] = load_yaml(project) if project.exists() else {}
    section = data.setdefault("environments", {})
    if name in section:
        typer.echo(f"El entorno '{name}' ya existe en {project}.", err=True)
        raise typer.Exit(code=1)

    section[name] = {"build_path": str(build_path) if build_path else None}
    project.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    typer.echo(f"Entorno '{name}' añadido a {project}.")


@env_app.command("launch")
def launch(
    env: str | None = typer.Option(
        None, "--env", help="Nombre de un entorno registrado (ver 'urc env list')."
    ),
    executable: Path | None = typer.Option(
        None,
        "--executable",
        "-e",
        help="Ruta al build de Unity. Si se omite, se usa la del entorno o se conecta al editor.",
    ),
    no_graphics: bool = typer.Option(
        False, "--no-graphics", help="Ejecuta el build sin ventana (modo headless)."
    ),
    worker_id: int | None = typer.Option(
        None,
        "--worker-id",
        help="ID de instancia paralela; usa uno distinto por cada build lanzado a la vez.",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Semilla aleatoria del entorno."),
    timeout: int | None = typer.Option(
        None, "--timeout", help="Segundos de espera para la conexión."
    ),
    project: Path = _PROJECT_OPTION,
) -> None:
    """Lanza (o conecta con) un entorno de Unity y verifica que el bridge por defecto funciona."""
    try:
        from urc.bridges.mlagents_bridge import MLAgentsBridge
    except ImportError as error:
        typer.echo(
            'El bridge "mlagents" no está instalado. Instálalo con: pip install "urc[mlagents]"',
            err=True,
        )
        raise typer.Exit(code=1) from error

    bridge_kwargs: dict[str, Any] = {}
    if env is not None:
        _load_registered_environments(project)
        try:
            spec = environment_registry.get(env)
        except KeyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
        bridge_kwargs.update(spec.bridge_options)
        if spec.build_path:
            bridge_kwargs["file_name"] = spec.build_path

    if executable is not None:
        bridge_kwargs["file_name"] = str(executable)
    if no_graphics:
        bridge_kwargs["no_graphics"] = True
    if worker_id is not None:
        bridge_kwargs["worker_id"] = worker_id
    if seed is not None:
        bridge_kwargs["seed"] = seed
    if timeout is not None:
        bridge_kwargs["timeout_wait"] = timeout

    if bridge_kwargs.get("file_name") is None:
        typer.echo("Esperando conexión con el editor de Unity (pulsa Play en la escena)...")
    else:
        typer.echo(f"Lanzando build: {bridge_kwargs['file_name']}")

    bridge = MLAgentsBridge(**bridge_kwargs)
    try:
        bridge.reset()
        obs_spec = bridge.observation_spec()
        action_spec = bridge.action_spec()
        typer.echo("Conexión establecida correctamente.")
        typer.echo(f"  Observaciones: shape={obs_spec.shape} dtype={obs_spec.dtype}")
        typer.echo(
            f"  Acciones:      shape={action_spec.shape} "
            f"dtype={action_spec.dtype} discreto={action_spec.discrete}"
        )
    finally:
        bridge.close()
