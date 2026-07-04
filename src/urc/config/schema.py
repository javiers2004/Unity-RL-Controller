from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = 500_000
    checkpoint_every: int = 50_000
    # Barra de progreso en vivo en la terminal (SB3, vía tqdm/rich). False por
    # defecto: coincide con el propio default de SB3 y evita ruido si la
    # salida no es una terminal interactiva (CI, logs redirigidos a archivo).
    progress_bar: bool = False


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["tensorboard", "wandb", "none"] = "tensorboard"
    # Nombre de proyecto para backends que lo necesitan (wandb.init(project=...)).
    project: str = "urc"


class LessonConfig(BaseModel):
    """Un escalón de un currículo: parámetros a aplicar y umbral para avanzar
    al siguiente. Ver `urc.algorithms.curriculum.CurriculumCallback`."""

    model_config = ConfigDict(extra="forbid")

    parameters: dict[str, Any] = Field(default_factory=dict)
    min_reward: float | None = None
    min_episodes: int = 1


class EnvironmentConfig(BaseModel):
    """Un entorno/mapa declarado en `urc.yaml` bajo `environments.<nombre>`."""

    model_config = ConfigDict(extra="forbid")

    build_path: str | None = None
    bridge_options: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    curriculum: list[LessonConfig] = Field(default_factory=list)


class UrcConfig(BaseModel):
    """Configuración resuelta de un experimento: qué bridge, algoritmo, entorno e
    hiperparámetros usar. `hyperparameters` se deja como dict libre porque su forma
    depende del algoritmo elegido, no hay un esquema único válido para todos."""

    model_config = ConfigDict(extra="forbid")

    bridge: str = "mlagents"
    bridge_options: dict[str, Any] = Field(default_factory=dict)
    algo: str = "sb3-ppo"
    env: str | None = None
    environments: dict[str, EnvironmentConfig] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output_dir: str = "runs"
