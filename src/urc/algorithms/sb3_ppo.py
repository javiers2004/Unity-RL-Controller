from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from urc.algorithms.gym_bridge import BridgeGymEnv
from urc.core.contracts import AlgorithmBackend, BridgeAdapter, EnvironmentSpec, Policy
from urc.core.registry import algorithms


class SB3Policy(Policy):
    def __init__(self, model: PPO) -> None:
        self._model = model

    def predict(self, observation: Any) -> Any:
        action, _ = self._model.predict(observation, deterministic=True)
        return action


@algorithms.register("sb3-ppo")
class SB3PPOBackend(AlgorithmBackend):
    """Backend por defecto: PPO de Stable-Baselines3 entrenado contra cualquier
    `BridgeAdapter` a través de `BridgeGymEnv`. Al pasar por el contrato genérico
    en vez de hablar directamente con Unity, funciona igual con el bridge de
    Unity, el de socket o el de subproceso — no está acoplado a Unity."""

    def train(
        self, bridge: BridgeAdapter, env_spec: EnvironmentSpec, config: dict[str, Any]
    ) -> Policy:
        env = BridgeGymEnv(bridge)
        hyperparameters = dict(config.get("hyperparameters", {}))
        training = config.get("training", {})
        resume_from = config.get("resume_from")

        if resume_from:
            model = PPO.load(resume_from, env=env)
            reset_num_timesteps = False
        else:
            model = PPO("MlpPolicy", env, **hyperparameters)
            reset_num_timesteps = True

        callback = None
        checkpoint_every = training.get("checkpoint_every")
        checkpoint_dir = config.get("checkpoint_dir")
        if checkpoint_every and checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            callback = CheckpointCallback(
                save_freq=checkpoint_every, save_path=str(checkpoint_dir), name_prefix="checkpoint"
            )

        model.learn(
            total_timesteps=training.get("max_steps", 500_000),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
        return SB3Policy(model)

    def load(self, checkpoint_path: str) -> Policy:
        return SB3Policy(PPO.load(checkpoint_path))
