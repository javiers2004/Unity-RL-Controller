import sys

import typer

from urc import __version__
from urc.cli.algo import algo_app
from urc.cli.compare import compare as compare_command
from urc.cli.config import config_app
from urc.cli.doctor import doctor as doctor_command
from urc.cli.env import env_app
from urc.cli.eval import eval_command
from urc.cli.init import init as init_command
from urc.cli.record import record as record_command
from urc.cli.train import train as train_command
from urc.cli.visualize import visualize as visualize_command

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


app.add_typer(env_app, name="env")
app.add_typer(config_app, name="config")
app.add_typer(algo_app, name="algo")
app.command("train")(train_command)
app.command("eval")(eval_command)
app.command("compare")(compare_command)
app.command("record")(record_command)
app.command("visualize")(visualize_command)
app.command("doctor")(doctor_command)
app.command("init")(init_command)


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
