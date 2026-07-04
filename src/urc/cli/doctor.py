from __future__ import annotations

import importlib
import sys

import typer

from urc import __version__


def doctor() -> None:
    """Diagnostica el entorno: Python, dependencias opcionales y GPU."""
    typer.echo(f"urc {__version__}")
    typer.echo(f"Python {sys.version.split()[0]} ({sys.executable})")
    typer.echo("")
    typer.echo("Dependencias opcionales:")
    _report_optional("mlagents_envs", "mlagents", "bridge 'mlagents' (conexión con Unity)")
    _report_optional("stable_baselines3", "sb3", "algoritmo 'sb3-ppo' (entrenamiento por defecto)")
    typer.echo("")
    _report_gpu()


def _report_optional(module_name: str, extra_name: str, description: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        typer.echo(f'  [ ] {description}: no instalado (pip install "urc[{extra_name}]")')
        return
    version = getattr(module, "__version__", "?")
    typer.echo(f"  [x] {description}: instalado ({module_name} {version})")


def _report_gpu() -> None:
    try:
        import torch
    except ImportError:
        typer.echo("GPU: no se puede comprobar (instala el extra 'sb3' para tener PyTorch)")
        return

    if torch.cuda.is_available():
        typer.echo(f"GPU: CUDA disponible ({torch.cuda.get_device_name(0)})")
    else:
        typer.echo(f"GPU: no disponible para PyTorch {torch.__version__} — se entrenará en CPU")
