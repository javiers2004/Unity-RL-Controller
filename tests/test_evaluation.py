from urc.core.contracts import BridgeAdapter, Policy, StepResult
from urc.core.evaluation import EvalResult, run_episodes


class ScriptedBridge(BridgeAdapter):
    """Episodios de longitud fija con una recompensa por episodio programada."""

    def __init__(self, episode_rewards: list[float], episode_length: int = 2) -> None:
        self._episode_rewards = episode_rewards
        self._episode_length = episode_length
        self._episode_index = -1
        self._step_in_episode = 0

    def reset(self):
        self._episode_index += 1
        self._step_in_episode = 0
        return [0.0]

    def step(self, action):
        self._step_in_episode += 1
        done = self._step_in_episode >= self._episode_length
        reward = self._episode_rewards[self._episode_index] / self._episode_length
        return StepResult(observation=[0.0], reward=reward, done=done)

    def observation_spec(self):
        raise NotImplementedError

    def action_spec(self):
        raise NotImplementedError

    def close(self):
        pass


class InfoSuccessBridge(ScriptedBridge):
    """Como ScriptedBridge, pero reporta info['success'] explícitamente."""

    def __init__(self, successes: list[bool]) -> None:
        super().__init__(episode_rewards=[0.0] * len(successes), episode_length=1)
        self._successes = successes

    def step(self, action):
        result = super().step(action)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info={"success": self._successes[self._episode_index]},
        )


class ConstantPolicy(Policy):
    def predict(self, observation):
        return 0


def test_run_episodes_counts_length_and_mean_reward():
    bridge = ScriptedBridge(episode_rewards=[2.0, 4.0], episode_length=2)

    result = run_episodes(bridge, ConstantPolicy(), num_episodes=2)

    assert len(result.episodes) == 2
    assert result.episodes[0].length == 2
    assert result.mean_reward == 3.0


def test_success_uses_info_success_when_present():
    bridge = InfoSuccessBridge(successes=[True, False, True])

    result = run_episodes(bridge, ConstantPolicy(), num_episodes=3)

    assert [e.success for e in result.episodes] == [True, False, True]
    assert result.success_rate == 2 / 3


def test_success_uses_threshold_when_no_info_success():
    bridge = ScriptedBridge(episode_rewards=[10.0, 1.0], episode_length=1)

    result = run_episodes(bridge, ConstantPolicy(), num_episodes=2, success_threshold=5.0)

    assert [e.success for e in result.episodes] == [True, False]
    assert result.success_rate == 0.5


def test_success_is_none_without_info_or_threshold():
    bridge = ScriptedBridge(episode_rewards=[1.0], episode_length=1)

    result = run_episodes(bridge, ConstantPolicy(), num_episodes=1)

    assert result.episodes[0].success is None
    assert result.success_rate is None


def test_truncates_episode_that_never_signals_done():
    class InfiniteBridge(BridgeAdapter):
        def reset(self):
            return [0.0]

        def step(self, action):
            return StepResult(observation=[0.0], reward=1.0, done=False)

        def observation_spec(self):
            raise NotImplementedError

        def action_spec(self):
            raise NotImplementedError

        def close(self):
            pass

    result = run_episodes(
        InfiniteBridge(), ConstantPolicy(), num_episodes=1, max_episode_steps=5
    )

    assert result.episodes[0].length == 5
    assert result.episodes[0].truncated is True
    assert result.episodes[0].success is None


def test_eval_result_to_dict_is_json_serializable_and_round_trips_stats():
    import json

    bridge = ScriptedBridge(episode_rewards=[2.0, 4.0], episode_length=1)
    result = run_episodes(bridge, ConstantPolicy(), num_episodes=2, checkpoint="ckpt.zip")

    data = json.loads(json.dumps(result.to_dict()))

    assert data["checkpoint"] == "ckpt.zip"
    assert data["mean_reward"] == result.mean_reward
    assert len(data["episodes"]) == 2


def test_eval_result_stats_are_zero_with_no_episodes():
    result = EvalResult(checkpoint="empty")

    assert result.mean_reward == 0.0
    assert result.std_reward == 0.0
    assert result.mean_length == 0.0
    assert result.success_rate is None
