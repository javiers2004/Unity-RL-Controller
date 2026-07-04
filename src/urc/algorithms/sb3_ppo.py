from __future__ import annotations

from stable_baselines3 import PPO

from urc.algorithms.sb3_base import SB3Backend
from urc.core.registry import algorithms


@algorithms.register("sb3-ppo")
class SB3PPOBackend(SB3Backend):
    """Backend por defecto: PPO de Stable-Baselines3 entrenado contra cualquier
    `BridgeAdapter` a través de `BridgeGymEnv`. Al pasar por el contrato genérico
    en vez de hablar directamente con Unity, funciona igual con el bridge de
    Unity, el de socket o el de subproceso — no está acoplado a Unity. Admite
    tanto acciones continuas como discretas."""

    algorithm_cls = PPO
