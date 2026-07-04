import pytest

pytest.importorskip("gymnasium")

from urc.algorithms.gym_bridge import BridgeGymEnv  # noqa: E402
from urc.core.contracts import ActionSpec, BridgeAdapter, ObservationSpec, StepResult  # noqa: E402


class FakeBridge(BridgeAdapter):
    def __init__(self, action_spec: ActionSpec) -> None:
        self._action_spec = action_spec
        self.closed = False
        self.last_action = None

    def reset(self):
        return [1.0, 2.0]

    def step(self, action):
        self.last_action = action
        return StepResult(observation=[3.0, 4.0], reward=1.5, done=True, info={"ok": True})

    def observation_spec(self):
        return ObservationSpec(shape=(2,))

    def action_spec(self):
        return self._action_spec

    def close(self):
        self.closed = True


def test_continuous_action_space_is_a_bounded_box():
    env = BridgeGymEnv(FakeBridge(ActionSpec(shape=(1,), discrete=False)))

    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_single_branch_discrete_action_space_is_discrete():
    import gymnasium as gym

    env = BridgeGymEnv(
        FakeBridge(ActionSpec(shape=(1,), discrete=True, discrete_branches=(3,)))
    )

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3


def test_multi_branch_discrete_action_space_is_multidiscrete():
    import gymnasium as gym

    env = BridgeGymEnv(
        FakeBridge(ActionSpec(shape=(2,), discrete=True, discrete_branches=(3, 4)))
    )

    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert list(env.action_space.nvec) == [3, 4]


def test_observation_space_matches_observation_spec_shape():
    env = BridgeGymEnv(FakeBridge(ActionSpec(shape=(1,), discrete=False)))

    assert env.observation_space.shape == (2,)


def test_reset_returns_observation_and_empty_info():
    env = BridgeGymEnv(FakeBridge(ActionSpec(shape=(1,), discrete=False)))

    observation, info = env.reset()

    assert list(observation) == [1.0, 2.0]
    assert info == {}


def test_step_maps_step_result_to_gym_tuple():
    bridge = FakeBridge(ActionSpec(shape=(1,), discrete=False))
    env = BridgeGymEnv(bridge)

    observation, reward, terminated, truncated, info = env.step([0.5])

    assert list(observation) == [3.0, 4.0]
    assert reward == 1.5
    assert terminated is True
    assert truncated is False
    assert info == {"ok": True}
    assert bridge.last_action == [0.5]


def test_close_closes_the_underlying_bridge():
    bridge = FakeBridge(ActionSpec(shape=(1,), discrete=False))
    env = BridgeGymEnv(bridge)

    env.close()

    assert bridge.closed is True
