from __future__ import annotations

from pathlib import Path

import typer
import yaml

from urc.config import ConfigError, diff_configs, overrides_to_dict, resolve_config

config_app = typer.Typer(
    help="Gestiona la configuración jerárquica (defaults, proyecto, experimento, overrides)."
)

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


@config_app.command("show")
def show(
    project: Path = _PROJECT_OPTION,
    experiment: Path | None = _EXPERIMENT_OPTION,
    set_: list[str] = _SET_OPTION,
) -> None:
    """Muestra la config final resuelta (defaults + proyecto + experimento + overrides)."""
    try:
        config = resolve_config(
            project_path=project,
            experiment_path=experiment,
            overrides=overrides_to_dict(set_),
        )
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(yaml.safe_dump(config.model_dump(), sort_keys=False, allow_unicode=True))


@config_app.command("validate")
def validate(
    project: Path = _PROJECT_OPTION,
    experiment: Path | None = _EXPERIMENT_OPTION,
    set_: list[str] = _SET_OPTION,
) -> None:
    """Valida que la configuración resuelta sea correcta, sin imprimirla."""
    try:
        resolve_config(
            project_path=project,
            experiment_path=experiment,
            overrides=overrides_to_dict(set_),
        )
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo("La configuración es válida.")


@config_app.command("diff")
def diff(
    a: Path = typer.Argument(..., help="Primer YAML (de proyecto o experimento) a comparar."),
    b: Path = typer.Argument(..., help="Segundo YAML (de proyecto o experimento) a comparar."),
) -> None:
    """Muestra las diferencias entre dos configuraciones ya resueltas contra los defaults."""
    try:
        config_a = resolve_config(project_path=a)
        config_b = resolve_config(project_path=b)
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    differences = diff_configs(config_a, config_b)
    if not differences:
        typer.echo("Sin diferencias.")
        return

    for path, (value_a, value_b) in differences.items():
        typer.echo(f"{path}: {value_a!r} -> {value_b!r}")
