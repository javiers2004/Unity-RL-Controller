import sys

import typer

from urc import __version__

# En Windows la consola no siempre usa UTF-8 por defecto, lo que corrompe
# acentos y el guion largo en la ayuda del CLI. Se fuerza aquí para que
# funcione igual en cualquier terminal sin que el usuario tenga que configurar nada.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

app = typer.Typer(
    name="urc",
    help=(
        "Unity RL Controller — controla entrenamientos de Reinforcement "
        "Learning en Unity desde la terminal."
    ),
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Unity RL Controller."""


@app.command()
def version() -> None:
    """Muestra la versión instalada de urc."""
    typer.echo(f"urc {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
