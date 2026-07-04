from __future__ import annotations

import json
from pathlib import Path

import typer

from urc.cli._shared import EXPERIMENT_OPTION, PROJECT_OPTION, SET_OPTION, load_bridge_and_policy
from urc.core.evaluation import run_episodes


def eval_command(
    checkpoint: Path = typer.Argument(..., help="Ruta al checkpoint a evaluar."),
    episodes: int = typer.Option(10, "--episodes", "-n", help="Número de episodios a evaluar."),
    success_threshold: float | None = typer.Option(
        None,
        "--success-threshold",
        help="Recompensa mínima del episodio para contar como éxito, si el "
        "entorno no reporta info['success'].",
    ),
    project: Path = PROJECT_OPTION,
    experiment: Path | None = EXPERIMENT_OPTION,
    set_: list[str] = SET_OPTION,
) -> None:
    """Evalúa un checkpoint N episodios y reporta reward medio, éxito y duración."""
    bridge, policy = load_bridge_and_policy(
        checkpoint, project=project, experiment=experiment, set_=set_
    )
    try:
        result = run_episodes(
            bridge,
            policy,
            episodes,
            success_threshold=success_threshold,
            checkpoint=str(checkpoint),
        )
    finally:
        bridge.close()

    typer.echo(f"Episodios:        {len(result.episodes)}")
    typer.echo(f"Reward medio:     {result.mean_reward:.3f} ± {result.std_reward:.3f}")
    typer.echo(f"Duración media:   {result.mean_length:.1f} pasos")
    success_rate = result.success_rate
    if success_rate is not None:
        typer.echo(f"Tasa de éxito:    {success_rate:.1%}")
    else:
        typer.echo("Tasa de éxito:    N/A (sin info['success'] ni --success-threshold)")

    output_path = checkpoint.with_name(f"eval_{checkpoint.stem}.json")
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    typer.echo(f"Resultados guardados en: {output_path}")
