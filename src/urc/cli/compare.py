from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer


def _resolve_eval_result_path(path: Path) -> Path:
    """Acepta un resultado de eval directamente, un checkpoint (busca su
    `eval_<nombre>.json` hermano) o una carpeta de run (usa el más reciente)."""
    if path.is_dir():
        candidates = sorted(path.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No hay resultados 'eval_*.json' en '{path}'.")
        return candidates[-1]

    if path.suffix == ".json":
        if not path.exists():
            raise FileNotFoundError(f"No existe '{path}'.")
        return path

    sibling = path.with_name(f"eval_{path.stem}.json")
    if not sibling.exists():
        raise FileNotFoundError(
            f"No existe '{sibling}'. Ejecuta 'urc eval {path}' primero."
        )
    return sibling


def _format_success_rate(success_rate: float | None) -> str:
    return f"{success_rate:.1%}" if success_rate is not None else "N/A"


def compare(
    paths: list[Path] = typer.Argument(
        ...,
        help="Resultados de eval (.json), checkpoints, o carpetas de run a comparar.",
    ),
) -> None:
    """Compara las métricas de `urc eval` entre dos o más runs/checkpoints."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        try:
            eval_path = _resolve_eval_result_path(path)
        except FileNotFoundError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=1) from error
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append((str(path), data))

    label_width = max(len(label) for label, _ in rows)
    header = (
        f"{'run':<{label_width}}  {'reward medio':>14}  {'± std':>8}  "
        f"{'éxito':>8}  {'pasos medios':>13}  {'episodios':>10}"
    )
    typer.echo(header)
    for label, data in rows:
        typer.echo(
            f"{label:<{label_width}}  {data['mean_reward']:>14.3f}  {data['std_reward']:>8.3f}  "
            f"{_format_success_rate(data['success_rate']):>8}  {data['mean_length']:>13.1f}  "
            f"{len(data['episodes']):>10}"
        )
