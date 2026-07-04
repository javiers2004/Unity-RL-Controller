from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from urc.core.contracts import BridgeAdapter


class CurriculumCallback(BaseCallback):
    """Avanza por las "lessons" de un currículo según la recompensa media de
    los últimos episodios.

    Cada lesson es un dict con `parameters` (aplicados a `bridge` vía
    `set_parameters` al entrar en la lesson) y `min_reward`/`min_episodes`
    (umbral para avanzar a la siguiente). Si el bridge no soporta
    `set_parameters` (implementación por defecto, no-op), el entrenamiento
    sigue funcionando igual, solo sin efecto real sobre el entorno.

    Solo sabe hablar con el bucle de entrenamiento de Stable-Baselines3 (lee
    `self.locals["rewards"]`/`["dones"]`, que rellena SB3 en cada `_on_step`);
    otros backends necesitarían su propia integración.
    """

    def __init__(
        self, bridge: BridgeAdapter, lessons: list[dict[str, Any]], verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self._bridge = bridge
        self._lessons = lessons
        self._lesson_index = 0
        self._episode_rewards: list[float] = []
        self._current_episode_reward = 0.0

    def _on_training_start(self) -> None:
        if self._lessons:
            self._bridge.set_parameters(self._lessons[0].get("parameters", {}))

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        self._current_episode_reward += float(rewards[0])
        if dones[0]:
            self._episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
            self._maybe_advance()

        return True

    def _maybe_advance(self) -> None:
        if self._lesson_index >= len(self._lessons) - 1:
            return

        lesson = self._lessons[self._lesson_index]
        min_reward = lesson.get("min_reward")
        min_episodes = lesson.get("min_episodes", 1)
        if min_reward is None or len(self._episode_rewards) < min_episodes:
            return

        recent = self._episode_rewards[-min_episodes:]
        if sum(recent) / len(recent) < min_reward:
            return

        self._lesson_index += 1
        next_lesson = self._lessons[self._lesson_index]
        self._bridge.set_parameters(next_lesson.get("parameters", {}))
        self._episode_rewards.clear()
