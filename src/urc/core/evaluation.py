from __future__ import annotations

import dataclasses
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from urc.core.contracts import BridgeAdapter, Policy


@dataclass
class EpisodeResult:
    reward: float
    length: int
    duration_seconds: float
    success: bool | None
    truncated: bool = False


@dataclass
class EvalResult:
    checkpoint: str
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return statistics.fmean(e.reward for e in self.episodes) if self.episodes else 0.0

    @property
    def std_reward(self) -> float:
        return statistics.pstdev(e.reward for e in self.episodes) if len(self.episodes) > 1 else 0.0

    @property
    def mean_length(self) -> float:
        return statistics.fmean(e.length for e in self.episodes) if self.episodes else 0.0

    @property
    def success_rate(self) -> float | None:
        successes = [e.success for e in self.episodes if e.success is not None]
        return (sum(successes) / len(successes)) if successes else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "episodes": [dataclasses.asdict(e) for e in self.episodes],
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_length": self.mean_length,
            "success_rate": self.success_rate,
        }


def _determine_success(
    total_reward: float, info: dict[str, Any], success_threshold: float | None
) -> bool | None:
    if "success" in info:
        return bool(info["success"])
    if success_threshold is not None:
        return total_reward >= success_threshold
    return None


def run_episodes(
    bridge: BridgeAdapter,
    policy: Policy,
    num_episodes: int,
    *,
    success_threshold: float | None = None,
    max_episode_steps: int = 10_000,
    checkpoint: str = "",
) -> EvalResult:
    """Corre `num_episodes` episodios completos con `policy` contra `bridge` y
    mide recompensa/duración/éxito por episodio. No sabe nada de Unity ni de
    ningún algoritmo concreto: solo usa el contrato BridgeAdapter/Policy, así
    que evalúa igual una política de SB3 que una de un plugin de terceros."""
    episodes: list[EpisodeResult] = []
    for _ in range(num_episodes):
        observation = bridge.reset()
        total_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}
        done = False
        truncated = False
        start = time.perf_counter()

        while not done:
            action = policy.predict(observation)
            result = bridge.step(action)
            observation = result.observation
            total_reward += result.reward
            info = result.info
            done = result.done
            steps += 1
            if not done and steps >= max_episode_steps:
                truncated = True
                break

        duration = time.perf_counter() - start
        success = None if truncated else _determine_success(total_reward, info, success_threshold)
        episodes.append(
            EpisodeResult(
                reward=total_reward,
                length=steps,
                duration_seconds=duration,
                success=success,
                truncated=truncated,
            )
        )

    return EvalResult(checkpoint=checkpoint, episodes=episodes)
