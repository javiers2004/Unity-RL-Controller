"""Tests de SB3SACBackend, análogos a test_sb3_ppo.py pero con SAC.

SAC necesita algunos timesteps de exploración pura (`learning_starts`) antes de
empezar a entrenar de verdad; se reduce a 1 para que la prueba siga siendo
rápida sin dejar de ejercitar el camino real de entrenamiento de SB3.
"""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from urc.algorithms.sb3_sac import SB3SACBackend  # noqa: E402
from urc.core.contracts import (  # noqa: E402
    ActionSpec,
    BridgeAdapter,
    EnvironmentSpec,
    ObservationSpec,
    StepResult,
)


class ContinuousTinyBridge(BridgeAdapter):
    def __init__(self) -> None:
        self._step_count = 0

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
        pass


class DiscreteTinyBridge(ContinuousTinyBridge):
    def action_spec(self):
        return ActionSpec(shape=(1,), discrete=True, discrete_branches=(3,))


TINY_TRAINING_CONFIG = {
    "hyperparameters": {"learning_starts": 1, "batch_size": 4, "train_freq": 1},
    "training": {"max_steps": 8},
}


def test_train_returns_a_policy_that_can_predict():
    backend = SB3SACBackend()
    env_spec = EnvironmentSpec(name="tiny")

    policy = backend.train(ContinuousTinyBridge(), env_spec, dict(TINY_TRAINING_CONFIG))

    assert policy.predict([0.0]) is not None


def test_train_rejects_discrete_action_spaces_with_a_clear_message():
    backend = SB3SACBackend()
    env_spec = EnvironmentSpec(name="tiny")

    with pytest.raises(ValueError, match="sb3-sac solo admite espacios de acciones continuos"):
        backend.train(DiscreteTinyBridge(), env_spec, dict(TINY_TRAINING_CONFIG))


def test_train_writes_checkpoints_when_configured(tmp_path: Path):
    backend = SB3SACBackend()
    env_spec = EnvironmentSpec(name="tiny")
    config = dict(TINY_TRAINING_CONFIG)
    config["training"] = {"max_steps": 8, "checkpoint_every": 4}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(ContinuousTinyBridge(), env_spec, config)

    assert len(list(tmp_path.glob("*.zip"))) >= 1


def test_load_returns_a_working_policy(tmp_path: Path):
    backend = SB3SACBackend()
    env_spec = EnvironmentSpec(name="tiny")
    config = dict(TINY_TRAINING_CONFIG)
    config["training"] = {"max_steps": 8, "checkpoint_every": 4}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(ContinuousTinyBridge(), env_spec, config)
    checkpoint = next(tmp_path.glob("*.zip"))

    assert backend.load(str(checkpoint)).predict([0.0]) is not None
