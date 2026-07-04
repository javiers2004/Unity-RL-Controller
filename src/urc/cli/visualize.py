from __future__ import annotations

import time
from pathlib import Path

import typer


def _launch_tensorboard(logdir: Path, port: int) -> str:
    """Lanza el servidor de TensorBoard en un hilo de fondo y devuelve su URL.

    Separado de `visualize()` para poder probarlo sin bloquear el proceso:
    `tb.launch()` arranca el servidor y vuelve inmediatamente, es el propio
    comando quien decide esperar (Ctrl+C) o no.
    """
    from tensorboard import program

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logdir), "--port", str(port)])
    return tb.launch()


def visualize(
    logdir: Path = typer.Argument(
        Path("runs"), help="Carpeta con los logs de TensorBoard (o la carpeta de un run concreto)."
    ),
    port: int = typer.Option(6006, "--port", help="Puerto del dashboard (0 = elegir uno libre)."),
) -> None:
    """Levanta TensorBoard apuntando a los logs de entrenamiento (`urc train`)."""
    try:
        url = _launch_tensorboard(logdir, port)
    except ImportError as error:
        typer.echo(
            'TensorBoard no está instalado. Instálalo con: pip install "urc[sb3]"', err=True
        )
        raise typer.Exit(code=1) from error

    typer.echo(f"TensorBoard disponible en: {url}")
    typer.echo("Ctrl+C para salir.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("Cerrando TensorBoard...")
