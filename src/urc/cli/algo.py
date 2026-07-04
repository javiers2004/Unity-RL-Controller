from __future__ import annotations

import typer

from urc.algorithms import algorithms as algorithm_registry
from urc.core.plugins import load_all_plugins

algo_app = typer.Typer(help="Consulta los algoritmos de entrenamiento registrados.")


@algo_app.command("list")
def list_algorithms() -> None:
    """Lista los nombres de algoritmos disponibles (built-in + plugins)."""
    load_all_plugins()
    for name in algorithm_registry.names():
        typer.echo(name)


@algo_app.command("info")
def info(name: str = typer.Argument(..., help="Nombre del algoritmo, p. ej. 'sb3-ppo'.")) -> None:
    """Muestra la descripción de un algoritmo registrado."""
    load_all_plugins()
    try:
        algorithm_cls = algorithm_registry.get(name)
    except (KeyError, ImportError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    description = (algorithm_cls.__doc__ or "(sin descripción)").strip()
    typer.echo(f"{name} ({algorithm_cls.__module__}.{algorithm_cls.__qualname__})")
    typer.echo("")
    typer.echo(description)
