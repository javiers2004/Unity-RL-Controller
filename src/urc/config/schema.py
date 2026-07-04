from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = 500_000
    checkpoint_every: int = 50_000


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["tensorboard", "wandb", "none"] = "tensorboard"


class UrcConfig(BaseModel):
    """Configuración resuelta de un experimento: qué bridge, algoritmo, entorno e
    hiperparámetros usar. `hyperparameters` se deja como dict libre porque su forma
    depende del algoritmo elegido, no hay un esquema único válido para todos."""

    model_config = ConfigDict(extra="forbid")

    bridge: str = "mlagents"
    algo: str = "mlagents-ppo"
    env: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
