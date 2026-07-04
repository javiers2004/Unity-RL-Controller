from __future__ import annotations

from pathlib import Path

import typer

_URC_YAML_TEMPLATE = """\
# Config del proyecto: sobreescribe aquí lo que quieras cambiar de los
# defaults de la librería. Ver `urc config show` para ver la config resuelta.
bridge: mlagents
"""


def init(name: str = typer.Argument(..., help="Nombre del nuevo proyecto.")) -> None:
    """Crea un proyecto nuevo con estructura y config por defecto."""
    project_dir = Path(name)
    if project_dir.exists() and any(project_dir.iterdir()):
        typer.echo(f"'{project_dir}' ya existe y no está vacío.", err=True)
        raise typer.Exit(code=1)

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "experiments").mkdir(exist_ok=True)
    (project_dir / "urc.yaml").write_text(_URC_YAML_TEMPLATE, encoding="utf-8")

    typer.echo(f"Proyecto creado en '{project_dir}'.")
    typer.echo(f"Siguiente paso: cd {name} && urc env launch")
