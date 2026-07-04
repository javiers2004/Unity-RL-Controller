from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class ActionSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"
    discrete: bool = False
    # Cardinalidad de cada rama discreta (p. ej. (3,) = una rama con 3 acciones
    # posibles). Solo tiene sentido si discrete=True; None en acciones continuas.
    # Necesario para construir un espacio Gym real (Discrete/MultiDiscrete), no
    # solo para mostrar la shape como en la Fase 3.
    discrete_branches: tuple[int, ...] | None = None


@dataclass
class StepResult:
    observation: Any
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentSpec:
    """Metadatos declarativos de un mapa/escena: describe el entorno, no contiene lógica."""

    name: str
    build_path: str | None = None
    observation_spec: ObservationSpec | None = None
    action_spec: ActionSpec | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    curriculum: dict[str, Any] | None = None


class BridgeAdapter(ABC):
    """Habla con el entorno de simulación (p. ej. Unity) e intercambia observaciones/acciones."""

    @abstractmethod
    def reset(self) -> Any:
        """Reinicia el episodio y devuelve la observación inicial."""

    @abstractmethod
    def step(self, action: Any) -> StepResult:
        """Aplica una acción y devuelve el resultado del paso."""

    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        """Describe la forma y tipo de las observaciones que produce el bridge."""

    @abstractmethod
    def action_spec(self) -> ActionSpec:
        """Describe la forma y tipo de las acciones que espera el bridge."""

    @abstractmethod
    def close(self) -> None:
        """Libera los recursos del bridge (procesos, sockets, builds de Unity...)."""


class Policy(ABC):
    """Política entrenada: decide una acción a partir de una observación."""

    @abstractmethod
    def predict(self, observation: Any) -> Any:
        """Devuelve la acción elegida para la observación dada."""


class AlgorithmBackend(ABC):
    """Implementación de un algoritmo de RL (PPO, SAC...) que entrena y carga políticas."""

    @abstractmethod
    def train(
        self, bridge: BridgeAdapter, env_spec: EnvironmentSpec, config: dict[str, Any]
    ) -> Policy:
        """Entrena una política contra `bridge` usando la configuración dada."""

    @abstractmethod
    def load(self, checkpoint_path: str) -> Policy:
        """Carga una política previamente entrenada desde un checkpoint."""
