from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import SAC

from urc.algorithms.gym_bridge import BridgeGymEnv
from urc.algorithms.sb3_base import SB3Backend
from urc.core.registry import algorithms


@algorithms.register("sb3-sac")
class SB3SACBackend(SB3Backend):
    """Backend alternativo a `sb3-ppo`: SAC de Stable-Baselines3.

    A diferencia de PPO, SAC solo admite espacios de acción continuos (no
    discretos ni multi-discretos) — es una limitación real del algoritmo, no
    del bridge ni del contrato, así que se comprueba explícitamente y se
    rechaza con un mensaje claro en vez de dejar que falle dentro de SB3."""

    algorithm_cls = SAC

    def _check_action_space(self, env: BridgeGymEnv) -> None:
        if isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            raise ValueError(
                "sb3-sac solo admite espacios de acciones continuos; "
                "usa sb3-ppo para entornos con acciones discretas."
            )
