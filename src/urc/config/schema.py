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


class RecordingConfig(BaseModel):
    """Vídeo automático del progreso de entrenamiento — ver
    `urc.algorithms.recording.RecordingCallback`. Solo tiene efecto con el
    bridge `mlagents` y el script `unity/UrcVideoRecorder/UrcVideoRecorder.cs`
    en la escena; con cualquier otro bridge, `enabled: true` solo avisa."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Velocidad de Time.timeScale durante la mayor parte del entrenamiento.
    fast_forward_speed: float = 20.0
    # Cada cuántos pasos se intercala un episodio a velocidad reducida
    # (normal_time_scale) — más bajo = más "épocas" distintas visibles en el
    # vídeo.
    normal_speed_every_n_steps: int = 500
    # Time.timeScale de esas ventanas periódicas. Por debajo de 1 a propósito
    # (4 veces más lento que "normal"): igual que en los episodios finales,
    # a timeScale=1 apenas da tiempo a capturar el movimiento completo de un
    # episodio corto.
    normal_time_scale: float = 0.25
    # Además de las ventanas periódicas de arriba, cada vez que la recompensa
    # media de los últimos `stabilization_window` episodios marca un nuevo
    # máximo (y han pasado al menos `min_episodes_between_breakthroughs`
    # episodios desde la última vez, sin superar `max_breakthroughs` en total),
    # se graba OTRA ventana a la velocidad más lenta (final_time_scale) — para
    # ver el momento exacto de cada mejora real, no solo el resultado final.
    # `max_breakthroughs` es necesario de verdad, no defensivo: en tareas que
    # convergen rápido (verificado con Basic) la recompensa marca un nuevo
    # máximo tan a menudo que, sin tope, la cámara lenta acababa dominando casi
    # todo el vídeo (4.017 fotogramas para solo 6.144 pasos entrenados).
    stabilization_window: int = 5
    min_episodes_between_breakthroughs: int = 20
    max_breakthroughs: int = 5
    # Episodios al terminar, con la política ya entrenada.
    final_episodes: int = 4
    # Time.timeScale de esos episodios finales. Por debajo de 1 a propósito
    # (4 veces más lento que tiempo real, igual que `normal_time_scale`): la
    # captura va a un ritmo real fijo (ver `fps`/CaptureIntervalSeconds), y
    # los episodios de ejemplo suelen ser cortos (Basic se resuelve en ~7
    # pasos, <1s real a timeScale=1) — a esa velocidad apenas da tiempo a
    # capturar el acercamiento y se ve como un teletransporte. 0.05 (20x más
    # lento) se probó primero pero alargaba demasiado esta fase si la
    # política final todavía no converge del todo (episodios más largos de
    # lo esperado) — 0.25 es un término medio razonable.
    final_time_scale: float = 0.25
    # Debe coincidir con CaptureIntervalSeconds en UrcVideoRecorder.cs (0.1s =
    # 10 fps): ScreenCapture.CaptureScreenshot solo admite una captura "en
    # vuelo" a la vez, así que Unity captura a un ritmo real fijo de 10 fps,
    # no a la tasa de refresco del editor — ver ROADMAP, Fase 8.
    fps: int = 10
    # Conserva los fotogramas PNG sueltos además del .mp4 (ocupan mucho más).
    keep_frames: bool = False


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
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    output_dir: str = "runs"
