"""Tests del wrapper MLAgentsBridge contra un UnityEnvironment falso.

No necesitan Unity real: sustituyen `UnityEnvironment` por un doble de prueba
que reproduce la parte de la API de mlagents_envs que usa el bridge (misma
forma de behavior_specs/get_steps/set_actions que la librería real). El
smoke test contra Unity de verdad se hace con `urc env launch`.
"""

import functools
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlagents_envs")

from urc.bridges.mlagents_bridge import MLAgentsBridge  # noqa: E402

BEHAVIOR_NAME = "Agent?team=0"


class FakeActionSpec:
    def __init__(self, continuous_size: int = 0, discrete_branches: tuple[int, ...] = ()) -> None:
        self.continuous_size = continuous_size
        self.discrete_branches = discrete_branches

    @property
    def discrete_size(self) -> int:
        return len(self.discrete_branches)

    def is_discrete(self) -> bool:
        return self.discrete_size > 0 and self.continuous_size == 0

    def is_continuous(self) -> bool:
        return self.discrete_size == 0 and self.continuous_size > 0


class FakeObservationSpec:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeUnityEnvironment:
    """Doble de UnityEnvironment: un behavior, un agente, episodios de 2 pasos."""

    def __init__(
        self,
        *,
        observation_shape: tuple[int, ...] = (2,),
        continuous_size: int = 1,
        discrete_branches: tuple[int, ...] = (),
        num_agents: int = 1,
        **_: object,
    ) -> None:
        self._behavior_spec = SimpleNamespace(
            observation_specs=[FakeObservationSpec(observation_shape)],
            action_spec=FakeActionSpec(continuous_size, discrete_branches),
        )
        self._num_agents = num_agents
        self.behavior_specs: dict[str, object] = {}
        self._decision_steps: dict[int, SimpleNamespace] = {}
        self._terminal_steps: dict[int, SimpleNamespace] = {}
        self._episode_step = 0
        self.last_action = None
        self.closed = False

    def reset(self) -> None:
        self.behavior_specs = {BEHAVIOR_NAME: self._behavior_spec}
        self._episode_step = 0
        shape = self._behavior_spec.observation_specs[0].shape
        self._decision_steps = {
            i: SimpleNamespace(obs=[np.zeros(shape)], reward=0.0) for i in range(self._num_agents)
        }
        self._terminal_steps = {}

    def set_actions(self, behavior_name: str, action_tuple: object) -> None:
        assert behavior_name == BEHAVIOR_NAME
        self.last_action = action_tuple

    def step(self) -> None:
        self._episode_step += 1
        shape = self._behavior_spec.observation_specs[0].shape
        obs = np.full(shape, self._episode_step, dtype=np.float32)
        if self._episode_step >= 2:
            self._decision_steps = {}
            self._terminal_steps = {0: SimpleNamespace(obs=[obs], reward=5.0)}
        else:
            self._decision_steps = {0: SimpleNamespace(obs=[obs], reward=1.0)}
            self._terminal_steps = {}

    def get_steps(self, behavior_name: str):
        assert behavior_name == BEHAVIOR_NAME
        return self._decision_steps, self._terminal_steps

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("urc.bridges.mlagents_bridge.UnityEnvironment", FakeUnityEnvironment)
    created = MLAgentsBridge()
    yield created
    created.close()


def test_reset_returns_initial_observation(bridge: MLAgentsBridge):
    obs = bridge.reset()
    assert list(obs) == [0.0, 0.0]


def test_step_returns_reward_and_not_done_before_terminal_step(bridge: MLAgentsBridge):
    bridge.reset()
    result = bridge.step(action=[0.5])
    assert result.reward == 1.0
    assert result.done is False


def test_step_marks_done_on_terminal_step(bridge: MLAgentsBridge):
    bridge.reset()
    bridge.step(action=[0.5])
    result = bridge.step(action=[0.5])
    assert result.reward == 5.0
    assert result.done is True


def test_observation_and_action_spec_for_continuous_behavior(bridge: MLAgentsBridge):
    bridge.reset()
    assert bridge.observation_spec().shape == (2,)

    action_spec = bridge.action_spec()
    assert action_spec.shape == (1,)
    assert action_spec.discrete is False


def test_close_closes_the_underlying_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("urc.bridges.mlagents_bridge.UnityEnvironment", FakeUnityEnvironment)
    created = MLAgentsBridge()
    created.close()
    assert created._env.closed is True


def test_discrete_action_spec_is_reported_correctly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "urc.bridges.mlagents_bridge.UnityEnvironment",
        functools.partial(FakeUnityEnvironment, continuous_size=0, discrete_branches=(3, 4)),
    )
    created = MLAgentsBridge()
    try:
        created.reset()
        action_spec = created.action_spec()
        assert action_spec.discrete is True
        assert action_spec.shape == (2,)
        assert action_spec.dtype == "int32"
        assert action_spec.discrete_branches == (3, 4)
    finally:
        created.close()


def test_multiple_active_agents_raise_not_implemented(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "urc.bridges.mlagents_bridge.UnityEnvironment",
        functools.partial(FakeUnityEnvironment, num_agents=2),
    )
    created = MLAgentsBridge()

    with pytest.raises(NotImplementedError):
        created.reset()

    created.close()
