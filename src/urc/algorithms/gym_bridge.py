from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from urc.core.contracts import ActionSpec, BridgeAdapter, ObservationSpec


class BridgeGymEnv(gym.Env):
    """Adapta cualquier `BridgeAdapter` a la interfaz estándar de Gymnasium.

    Así cualquier librería del ecosistema Gym (Stable-Baselines3, RLlib...) puede
    entrenar contra el bridge sin saber nada de Unity ni de nuestro contrato: el
    bridge de Unity, el de socket o el de subproceso son indistinguibles aquí.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, bridge: BridgeAdapter) -> None:
        super().__init__()
        self._bridge = bridge
        self.observation_space = _observation_space(bridge.observation_spec())
        self.action_space = _action_space(bridge.action_spec())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        observation = np.asarray(self._bridge.reset(), dtype=np.float32)
        return observation, {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self._bridge.step(action)
        observation = np.asarray(result.observation, dtype=np.float32)
        truncated = False
        return observation, float(result.reward), result.done, truncated, result.info

    def close(self) -> None:
        self._bridge.close()


def _observation_space(spec: ObservationSpec) -> gym.Space:
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=spec.shape, dtype=np.float32)


def _action_space(spec: ActionSpec) -> gym.Space:
    if not spec.discrete:
        # SB3 exige límites finitos en un Box de acciones (a diferencia del de
        # observaciones). [-1, 1] es además la convención habitual de ML-Agents
        # para acciones continuas.
        return gym.spaces.Box(low=-1.0, high=1.0, shape=spec.shape, dtype=np.float32)

    branches = spec.discrete_branches or spec.shape
    if len(branches) == 1:
        return gym.spaces.Discrete(int(branches[0]))
    return gym.spaces.MultiDiscrete(list(branches))
