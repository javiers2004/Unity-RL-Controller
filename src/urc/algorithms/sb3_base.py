from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback

from urc.algorithms.curriculum import CurriculumCallback
from urc.algorithms.gym_bridge import BridgeGymEnv
from urc.core.contracts import AlgorithmBackend, BridgeAdapter, EnvironmentSpec, Policy


class SB3Policy(Policy):
    def __init__(self, model: BaseAlgorithm) -> None:
        self._model = model

    def predict(self, observation: Any) -> Any:
        action, _ = self._model.predict(observation, deterministic=True)
        return action


class SB3Backend(AlgorithmBackend):
    """Base compartida por los backends de Stable-Baselines3 (PPO, SAC...).

    Todos entrenan contra cualquier `BridgeAdapter` a través de `BridgeGymEnv`
    (no acoplados a Unity), con la misma mecánica de entrenar/reanudar/
    checkpoint. Lo único que cambia entre algoritmos concretos es la clase de
    SB3 a usar y, opcionalmente, qué espacios de acción soporta.
    """

    algorithm_cls: ClassVar[type[BaseAlgorithm]]
    policy_name: ClassVar[str] = "MlpPolicy"

    def train(
        self, bridge: BridgeAdapter, env_spec: EnvironmentSpec, config: dict[str, Any]
    ) -> Policy:
        env = BridgeGymEnv(bridge)
        self._check_action_space(env)

        if env_spec.parameters:
            bridge.set_parameters(env_spec.parameters)

        hyperparameters = dict(config.get("hyperparameters", {}))
        training = config.get("training", {})
        resume_from = config.get("resume_from")

        if resume_from:
            model = self.algorithm_cls.load(resume_from, env=env)
            reset_num_timesteps = False
        else:
            model = self.algorithm_cls(self.policy_name, env, **hyperparameters)
            reset_num_timesteps = True

        callbacks = []
        checkpoint_every = training.get("checkpoint_every")
        checkpoint_dir = config.get("checkpoint_dir")
        if checkpoint_every and checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_every,
                    save_path=str(checkpoint_dir),
                    name_prefix="checkpoint",
                )
            )
        if env_spec.curriculum:
            callbacks.append(CurriculumCallback(bridge, env_spec.curriculum))

        model.learn(
            total_timesteps=training.get("max_steps", 500_000),
            callback=callbacks or None,
            reset_num_timesteps=reset_num_timesteps,
        )
        return SB3Policy(model)

    def load(self, checkpoint_path: str) -> Policy:
        return SB3Policy(self.algorithm_cls.load(checkpoint_path))

    def _check_action_space(self, env: BridgeGymEnv) -> None:
        """Hook para que las subclases rechacen espacios de acción que su
        algoritmo no soporta (p. ej. SAC no admite acciones discretas)."""
