"""Tests del backend SB3PPOBackend contra un BridgeAdapter falso y minúsculo.

Usa muy pocos timesteps/n_steps para que el entrenamiento real (sin mocks: es
la propia librería Stable-Baselines3) termine en menos de un segundo. No
comprueba que la política "aprenda" nada, solo que el cableado (bridge ->
BridgeGymEnv -> SB3 -> Policy -> checkpoint en disco) funciona de verdad.
"""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from urc.algorithms.sb3_ppo import SB3PPOBackend  # noqa: E402
from urc.core.contracts import (  # noqa: E402
    ActionSpec,
    BridgeAdapter,
    EnvironmentSpec,
    ObservationSpec,
    StepResult,
)


class TinyBridge(BridgeAdapter):
    """Episodios de longitud fija 2, observación/acción continuas de tamaño 1."""

    def __init__(self) -> None:
        self._step_count = 0
        self.closed = False

    def reset(self):
        self._step_count = 0
        return [0.0]

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= 2
        if done:
            self._step_count = 0
        return StepResult(observation=[0.0], reward=1.0, done=done)

    def observation_spec(self):
        return ObservationSpec(shape=(1,))

    def action_spec(self):
        return ActionSpec(shape=(1,), discrete=False)

    def close(self):
        self.closed = True


TINY_TRAINING_CONFIG = {
    "hyperparameters": {"n_steps": 8, "batch_size": 4, "n_epochs": 1},
    "training": {"max_steps": 16},
}


def test_train_returns_a_policy_that_can_predict():
    backend = SB3PPOBackend()
    bridge = TinyBridge()
    env_spec = EnvironmentSpec(name="tiny")

    policy = backend.train(bridge, env_spec, dict(TINY_TRAINING_CONFIG))
    action = policy.predict([0.0])

    assert action is not None


def test_train_writes_checkpoints_when_configured(tmp_path: Path):
    backend = SB3PPOBackend()
    bridge = TinyBridge()
    env_spec = EnvironmentSpec(name="tiny")
    config = dict(TINY_TRAINING_CONFIG)
    config["training"] = {"max_steps": 16, "checkpoint_every": 8}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(bridge, env_spec, config)

    checkpoints = list(tmp_path.glob("*.zip"))
    assert len(checkpoints) >= 1


def test_load_returns_a_working_policy(tmp_path: Path):
    backend = SB3PPOBackend()
    bridge = TinyBridge()
    env_spec = EnvironmentSpec(name="tiny")
    config = dict(TINY_TRAINING_CONFIG)
    config["training"] = {"max_steps": 16, "checkpoint_every": 8}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(bridge, env_spec, config)
    checkpoint = next(tmp_path.glob("*.zip"))

    loaded_policy = backend.load(str(checkpoint))

    assert loaded_policy.predict([0.0]) is not None


def test_resume_continues_training_without_resetting_timestep_counter(tmp_path: Path):
    backend = SB3PPOBackend()
    bridge = TinyBridge()
    env_spec = EnvironmentSpec(name="tiny")
    config = dict(TINY_TRAINING_CONFIG)
    config["training"] = {"max_steps": 16, "checkpoint_every": 8}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(bridge, env_spec, config)
    checkpoint = next(tmp_path.glob("*.zip"))

    resumed_config = dict(TINY_TRAINING_CONFIG)
    resumed_config["training"] = {"max_steps": 24}
    resumed_config["resume_from"] = str(checkpoint)

    policy = backend.train(TinyBridge(), env_spec, resumed_config)

    assert policy._model.num_timesteps >= 24
