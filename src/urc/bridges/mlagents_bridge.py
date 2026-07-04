from __future__ import annotations

from typing import Any

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from urc.core.contracts import ActionSpec, BridgeAdapter, ObservationSpec, StepResult
from urc.core.registry import bridges


@bridges.register("mlagents")
class MLAgentsBridge(BridgeAdapter):
    """Bridge por defecto: envuelve `mlagents_envs.UnityEnvironment`.

    Soporta, por ahora, un único behavior con un único agente activo a la vez
    y un único sensor de observación: el caso de la mayoría de entornos de
    ejemplo de ML-Agents (Basic, GridWorld...). Multi-agente/multi-behavior
    queda para cuando el contrato lo necesite explícitamente (no se adivina
    aquí para no comprometerse con un diseño no probado contra Unity real).
    """

    def __init__(
        self,
        file_name: str | None = None,
        *,
        worker_id: int = 0,
        base_port: int | None = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
    ) -> None:
        self._env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            no_graphics=no_graphics,
            timeout_wait=timeout_wait,
        )

    def reset(self) -> Any:
        self._env.reset()
        return self._only_decision_step().obs[0]

    def step(self, action: Any) -> StepResult:
        behavior_name = self._only_behavior_name()
        action_spec = self._env.behavior_specs[behavior_name].action_spec

        if action_spec.is_discrete():
            action_tuple = ActionTuple(discrete=self._as_batch(action, np.int32))
        else:
            action_tuple = ActionTuple(continuous=self._as_batch(action, np.float32))

        self._env.set_actions(behavior_name, action_tuple)
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(behavior_name)
        if len(terminal_steps) == 1:
            step = next(iter(terminal_steps.values()))
            return StepResult(observation=step.obs[0], reward=step.reward, done=True)
        if len(decision_steps) == 1:
            step = next(iter(decision_steps.values()))
            return StepResult(observation=step.obs[0], reward=step.reward, done=False)

        raise NotImplementedError(
            "MLAgentsBridge solo soporta un agente activo a la vez; se encontraron "
            f"{len(decision_steps)} en decisión y {len(terminal_steps)} terminados."
        )

    def observation_spec(self) -> ObservationSpec:
        specs = self._env.behavior_specs[self._only_behavior_name()].observation_specs
        if len(specs) != 1:
            raise NotImplementedError(
                "MLAgentsBridge solo soporta un sensor de observación; "
                f"el entorno expone {len(specs)}."
            )
        return ObservationSpec(shape=tuple(specs[0].shape))

    def action_spec(self) -> ActionSpec:
        spec = self._env.behavior_specs[self._only_behavior_name()].action_spec
        if spec.is_discrete():
            return ActionSpec(shape=(spec.discrete_size,), dtype="int32", discrete=True)
        return ActionSpec(shape=(spec.continuous_size,), discrete=False)

    def close(self) -> None:
        self._env.close()

    def _only_behavior_name(self) -> str:
        if not self._env.behavior_specs:
            self._env.reset()

        behavior_names = list(self._env.behavior_specs.keys())
        if len(behavior_names) != 1:
            raise NotImplementedError(
                "MLAgentsBridge solo soporta un behavior por entorno; se encontraron "
                f"{len(behavior_names)}: {behavior_names}"
            )
        return behavior_names[0]

    def _only_decision_step(self) -> Any:
        behavior_name = self._only_behavior_name()
        decision_steps, _ = self._env.get_steps(behavior_name)
        if len(decision_steps) != 1:
            raise NotImplementedError(
                "MLAgentsBridge solo soporta un agente activo a la vez; se encontraron "
                f"{len(decision_steps)} tras el reset."
            )
        return next(iter(decision_steps.values()))

    @staticmethod
    def _as_batch(action: Any, dtype: type) -> np.ndarray:
        array = np.atleast_1d(np.asarray(action, dtype=dtype))
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array
