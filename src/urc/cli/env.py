from __future__ import annotations

from pathlib import Path

import typer

env_app = typer.Typer(
    help="Gestiona el entorno/mapa de Unity: lanzar builds y verificar la conexión."
)


@env_app.command("launch")
def launch(
    executable: Path | None = typer.Option(
        None,
        "--executable",
        "-e",
        help="Ruta al build headless de Unity. Si se omite, se conecta al editor abierto.",
    ),
    no_graphics: bool = typer.Option(
        False, "--no-graphics", help="Ejecuta el build sin ventana (modo headless)."
    ),
    worker_id: int = typer.Option(
        0,
        "--worker-id",
        help="ID de instancia paralela; usa uno distinto por cada build lanzado a la vez.",
    ),
    seed: int = typer.Option(0, "--seed", help="Semilla aleatoria del entorno."),
    timeout: int = typer.Option(60, "--timeout", help="Segundos de espera para la conexión."),
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

    if executable is None:
        typer.echo("Esperando conexión con el editor de Unity (pulsa Play en la escena)...")
    else:
        typer.echo(f"Lanzando build: {executable}")

    bridge = MLAgentsBridge(
        file_name=str(executable) if executable else None,
        no_graphics=no_graphics,
        worker_id=worker_id,
        seed=seed,
        timeout_wait=timeout,
    )
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
