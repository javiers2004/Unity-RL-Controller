"""Tests de la integración de logging (TensorBoard/wandb) en SB3Backend.

Los de TensorBoard entrenan de verdad unos pocos timesteps y comprueban que
aparecen archivos de eventos reales en disco, no que se llamó a un mock. El
de wandb usa WANDB_MODE=disabled (variable que la propia librería respeta)
para probar el cableado real sin red ni credenciales.
"""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from urc.algorithms.sb3_base import _build_logging  # noqa: E402
from urc.algorithms.sb3_ppo import SB3PPOBackend  # noqa: E402
from urc.core.contracts import (  # noqa: E402
    ActionSpec,
    BridgeAdapter,
    EnvironmentSpec,
    ObservationSpec,
    StepResult,
)


class TinyBridge(BridgeAdapter):
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


TINY_TRAINING_CONFIG = {
    "hyperparameters": {"n_steps": 8, "batch_size": 4, "n_epochs": 1},
    "training": {"max_steps": 16},
}


def test_build_logging_returns_none_for_backend_none():
    tensorboard_log, callbacks = _build_logging({"backend": "none"}, {}, "/tmp/whatever")

    assert tensorboard_log is None
    assert callbacks == []


def test_build_logging_sets_tensorboard_path_for_tensorboard_backend():
    tensorboard_log, callbacks = _build_logging(
        {"backend": "tensorboard"}, {}, "/tmp/my-run"
    )

    assert tensorboard_log == str(Path("/tmp/my-run") / "tensorboard")
    assert callbacks == []


def test_build_logging_without_checkpoint_dir_disables_tensorboard():
    tensorboard_log, _ = _build_logging({"backend": "tensorboard"}, {}, None)

    assert tensorboard_log is None


def test_build_logging_wandb_backend_also_enables_tensorboard_sync(monkeypatch, tmp_path: Path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    tensorboard_log, callbacks = _build_logging(
        {"backend": "wandb", "project": "urc-test"}, {"learning_rate": 1e-3}, str(tmp_path)
    )

    assert tensorboard_log == str(tmp_path / "tensorboard")
    assert len(callbacks) == 1


def test_train_writes_real_tensorboard_event_files(tmp_path: Path):
    backend = SB3PPOBackend()
    config = dict(TINY_TRAINING_CONFIG)
    config["logging"] = {"backend": "tensorboard"}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(TinyBridge(), EnvironmentSpec(name="tiny"), config)

    event_files = list(tmp_path.glob("tensorboard/**/events.out.tfevents.*"))
    assert len(event_files) >= 1


def test_train_writes_no_tensorboard_files_when_backend_is_none(tmp_path: Path):
    backend = SB3PPOBackend()
    config = dict(TINY_TRAINING_CONFIG)
    config["logging"] = {"backend": "none"}
    config["checkpoint_dir"] = str(tmp_path)

    backend.train(TinyBridge(), EnvironmentSpec(name="tiny"), config)

    assert not (tmp_path / "tensorboard").exists()


def test_train_with_progress_bar_enabled_still_completes(tmp_path: Path):
    backend = SB3PPOBackend()
    config = dict(TINY_TRAINING_CONFIG)
    config["training"] = {"max_steps": 16, "progress_bar": True}

    policy = backend.train(TinyBridge(), EnvironmentSpec(name="tiny"), config)

    assert policy.predict([0.0]) is not None


def test_train_wandb_backend_end_to_end_in_disabled_mode(monkeypatch, tmp_path: Path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    backend = SB3PPOBackend()
    config = dict(TINY_TRAINING_CONFIG)
    config["logging"] = {"backend": "wandb", "project": "urc-test"}
    config["checkpoint_dir"] = str(tmp_path)

    policy = backend.train(TinyBridge(), EnvironmentSpec(name="tiny"), config)

    assert policy.predict([0.0]) is not None
    assert list(tmp_path.glob("tensorboard/**/events.out.tfevents.*"))


def test_train_wandb_backend_without_wandb_installed_raises_friendly_error(
    monkeypatch, tmp_path: Path
):
    import sys

    # `sys.modules[name] = None` es el mecanismo estándar de Python para
    # forzar ImportError en `import name`, sin tocar el import global de
    # ningún otro módulo (a diferencia de parchear builtins.__import__).
    monkeypatch.setitem(sys.modules, "wandb", None)

    backend = SB3PPOBackend()
    config = dict(TINY_TRAINING_CONFIG)
    config["logging"] = {"backend": "wandb"}
    config["checkpoint_dir"] = str(tmp_path)

    with pytest.raises(ImportError, match='pip install "urc\\[wandb\\]"'):
        backend.train(TinyBridge(), EnvironmentSpec(name="tiny"), config)
