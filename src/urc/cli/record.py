from __future__ import annotations

import json
from pathlib import Path

import typer

from urc.cli._shared import EXPERIMENT_OPTION, PROJECT_OPTION, SET_OPTION, load_bridge_and_policy
from urc.core.jsonutil import json_safe


def record(
    checkpoint: Path = typer.Argument(..., help="Ruta al checkpoint con el que grabar."),
    episodes: int = typer.Option(1, "--episodes", "-n", help="Número de episodios a grabar."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Ruta del archivo .jsonl de salida (por defecto, junto al checkpoint).",
    ),
    max_episode_steps: int = typer.Option(
        10_000, "--max-episode-steps", help="Corta el episodio si nunca llega a terminar."
    ),
    project: Path = PROJECT_OPTION,
    experiment: Path | None = EXPERIMENT_OPTION,
    set_: list[str] = SET_OPTION,
) -> None:
    """Graba la trayectoria (observación/acción/recompensa por paso) de N episodios.

    No es un vídeo de píxeles: Unity no expone su renderizado por este canal.
    Es un replay estructurado — útil para depurar o analizar el
    comportamiento de una política sin necesitar Unity abierto para verlo.
    """
    bridge, policy = load_bridge_and_policy(
        checkpoint, project=project, experiment=experiment, set_=set_
    )
    output_path = output or checkpoint.with_name(f"replay_{checkpoint.stem}.jsonl")

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for episode_index in range(episodes):
                observation = bridge.reset()
                done = False
                step_index = 0
                while not done and step_index < max_episode_steps:
                    action = policy.predict(observation)
                    result = bridge.step(action)
                    handle.write(
                        json.dumps(
                            {
                                "episode": episode_index,
                                "step": step_index,
                                "observation": json_safe(observation),
                                "action": json_safe(action),
                                "reward": result.reward,
                                "done": result.done,
                            }
                        )
                        + "\n"
                    )
                    observation = result.observation
                    done = result.done
                    step_index += 1
    finally:
        bridge.close()

    typer.echo(f"Replay guardado en: {output_path} ({episodes} episodio(s)).")
