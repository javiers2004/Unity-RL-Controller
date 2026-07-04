"""Tests de CurriculumCallback contra un bridge falso y el bucle de SB3 simulado
a mano (self.locals), sin necesitar entrenar de verdad ni tener stable-baselines3
instalado: BaseCallback es la única pieza de SB3 que se usa, y solo por herencia.
"""

import pytest

pytest.importorskip("stable_baselines3")

from urc.algorithms.curriculum import CurriculumCallback  # noqa: E402
from urc.core.contracts import BridgeAdapter  # noqa: E402


class RecordingBridge(BridgeAdapter):
    def __init__(self) -> None:
        self.received_parameters: list[dict] = []

    def reset(self):
        return None

    def step(self, action):
        raise NotImplementedError

    def observation_spec(self):
        raise NotImplementedError

    def action_spec(self):
        raise NotImplementedError

    def close(self):
        pass

    def set_parameters(self, parameters):
        self.received_parameters.append(dict(parameters))


LESSONS = [
    {"parameters": {"difficulty": 0.1}, "min_reward": 1.0, "min_episodes": 2},
    {"parameters": {"difficulty": 0.5}, "min_reward": 2.0, "min_episodes": 1},
    {"parameters": {"difficulty": 0.9}},
]


def _make_callback(bridge: RecordingBridge, lessons: list[dict]) -> CurriculumCallback:
    callback = CurriculumCallback(bridge, lessons)
    callback.locals = {}
    return callback


def _step(callback: CurriculumCallback, reward: float, done: bool) -> None:
    callback.locals = {"rewards": [reward], "dones": [done]}
    callback._on_step()


def test_applies_first_lesson_parameters_on_training_start():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, LESSONS)

    callback._on_training_start()

    assert bridge.received_parameters == [{"difficulty": 0.1}]


def test_does_not_advance_before_min_episodes_reached():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, LESSONS)
    callback._on_training_start()

    _step(callback, reward=1.5, done=True)  # 1 episodio completo, hacen falta 2

    assert bridge.received_parameters == [{"difficulty": 0.1}]


def test_does_not_advance_below_min_reward():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, LESSONS)
    callback._on_training_start()

    _step(callback, reward=0.1, done=True)
    _step(callback, reward=0.1, done=True)

    assert bridge.received_parameters == [{"difficulty": 0.1}]


def test_advances_to_next_lesson_once_threshold_reached():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, LESSONS)
    callback._on_training_start()

    _step(callback, reward=1.0, done=True)
    _step(callback, reward=1.0, done=True)

    assert bridge.received_parameters == [{"difficulty": 0.1}, {"difficulty": 0.5}]


def test_accumulates_reward_across_steps_within_one_episode():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, LESSONS)
    callback._on_training_start()

    # 2 episodios de 0.6 + 0.6 = 1.2 >= min_reward 1.0, con min_episodes 2
    _step(callback, reward=0.6, done=False)
    _step(callback, reward=0.6, done=True)
    _step(callback, reward=0.6, done=False)
    _step(callback, reward=0.6, done=True)

    assert bridge.received_parameters == [{"difficulty": 0.1}, {"difficulty": 0.5}]


def test_does_not_advance_past_the_last_lesson():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, [{"parameters": {"difficulty": 1.0}}])
    callback._on_training_start()

    _step(callback, reward=1000.0, done=True)
    _step(callback, reward=1000.0, done=True)

    assert bridge.received_parameters == [{"difficulty": 1.0}]


def test_empty_lessons_list_does_nothing():
    bridge = RecordingBridge()
    callback = _make_callback(bridge, [])

    callback._on_training_start()
    _step(callback, reward=1.0, done=True)

    assert bridge.received_parameters == []
