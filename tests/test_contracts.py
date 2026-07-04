import pytest

from urc.core.contracts import (
    ActionSpec,
    AlgorithmBackend,
    BridgeAdapter,
    ObservationSpec,
    Policy,
    StepResult,
)


class NoOpBridge(BridgeAdapter):
    def reset(self):
        return None

    def step(self, action):
        return StepResult(observation=None, reward=0.0, done=True)

    def observation_spec(self):
        return ObservationSpec(shape=(1,))

    def action_spec(self):
        return ActionSpec(shape=(1,))

    def close(self):
        pass


class NoOpPolicy(Policy):
    def predict(self, observation):
        return None


class NoOpAlgorithm(AlgorithmBackend):
    def train(self, bridge, env_spec, config):
        return NoOpPolicy()

    def load(self, checkpoint_path):
        return NoOpPolicy()


def test_bridge_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BridgeAdapter()


def test_incomplete_bridge_subclass_cannot_be_instantiated():
    class IncompleteBridge(BridgeAdapter):
        def reset(self):
            return None

    with pytest.raises(TypeError):
        IncompleteBridge()


def test_noop_bridge_satisfies_the_contract():
    bridge = NoOpBridge()

    bridge.reset()
    result = bridge.step(action=None)
    bridge.close()

    assert result.done is True
    assert result.info == {}


def test_noop_algorithm_satisfies_the_contract():
    algorithm = NoOpAlgorithm()
    bridge = NoOpBridge()

    policy = algorithm.train(bridge, env_spec=None, config={})
    assert policy.predict(observation=None) is None

    loaded = algorithm.load("checkpoint.bin")
    assert isinstance(loaded, Policy)
