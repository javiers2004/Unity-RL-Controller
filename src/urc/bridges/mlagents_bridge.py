from __future__ import annotations

from typing import Any

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)

from urc.bridges._recording_channel import RecordingControlChannel
from urc.core.contracts import ActionSpec, BridgeAdapter, ObservationSpec, StepResult
from urc.core.registry import bridges


@bridges.register("mlagents")
class MLAgentsBridge(BridgeAdapter):
    """Bridge por defecto: envuelve `mlagents_envs.UnityEnvironment`.

    Soporta, por ahora, un único behavior con un único agente activo a la vez.
    Si el entorno expone varios sensores de observación (p. ej. un
    `RayPerceptionSensor` para detectar obstáculos además del vector base —
    el caso de WallJump, verificado contra Unity real), se concatenan en un
    único vector plano: `urc` solo entrena con políticas `MlpPolicy`, así que
    no hace falta preservar la estructura por sensor (eso importaría para una
    política con CNN sobre observaciones visuales, fuera de alcance por
    ahora). Multi-agente/multi-behavior queda para cuando el contrato lo
    necesite explícitamente (no se adivina aquí para no comprometerse con un
    diseño no probado contra Unity real).
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
        self._parameters_channel = EnvironmentParametersChannel()
        self._recording_channel = RecordingControlChannel()
        self._env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            base_port=base_port,
            seed=seed,
            no_graphics=no_graphics,
            timeout_wait=timeout_wait,
            side_channels=[self._parameters_channel, self._recording_channel],
        )

    def reset(self) -> Any:
        self._env.reset()
        return self._flatten_observation(self._only_decision_step().obs)

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
            return StepResult(
                observation=self._flatten_observation(step.obs),
                reward=float(step.reward),
                done=True,
            )
        if len(decision_steps) == 1:
            step = next(iter(decision_steps.values()))
            return StepResult(
                observation=self._flatten_observation(step.obs),
                reward=float(step.reward),
                done=False,
            )

        raise NotImplementedError(
            "MLAgentsBridge solo soporta un agente activo a la vez; se encontraron "
            f"{len(decision_steps)} en decisión y {len(terminal_steps)} terminados."
        )

    def observation_spec(self) -> ObservationSpec:
        specs = self._env.behavior_specs[self._only_behavior_name()].observation_specs
        total_size = sum(int(np.prod(spec.shape)) for spec in specs)
        return ObservationSpec(shape=(total_size,))

    def action_spec(self) -> ActionSpec:
        spec = self._env.behavior_specs[self._only_behavior_name()].action_spec
        if spec.is_discrete():
            return ActionSpec(
                shape=(spec.discrete_size,),
                dtype="int32",
                discrete=True,
                discrete_branches=tuple(spec.discrete_branches),
            )
        return ActionSpec(shape=(spec.continuous_size,), discrete=False)

    def close(self) -> None:
        self._env.close()

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        """Envía parámetros float a Unity vía `EnvironmentParametersChannel`.

        La escena solo los usa si su propio código C# los lee explícitamente
        con `Academy.Instance.EnvironmentParameters.GetWithDefault(...)` — el
        envío en sí funciona igual lo lea la escena o no.
        """
        for key, value in parameters.items():
            self._parameters_channel.set_float_parameter(key, float(value))

    def start_recording(self, output_dir: str) -> None:
        """Le dice a la escena (vía `unity/UrcVideoRecorder/UrcVideoRecorder.cs`) que
        empiece a capturar fotogramas en `output_dir`. No hace nada si la escena no
        tiene ese script — es responsabilidad de quien active `recording.enabled`.
        """
        self._recording_channel.start_recording(output_dir)

    def stop_recording(self) -> None:
        """Le dice a la escena que deje de capturar fotogramas. Hay que llamarlo
        antes de tocar la carpeta de fotogramas desde Python (p. ej. para
        ensamblar el vídeo y borrarlos): la captura en Unity es una corrutina
        en bucle infinito que, si no se le avisa, sigue intentando escribir
        indefinidamente aunque Python ya haya terminado."""
        self._recording_channel.stop_recording()

    def set_time_scale(self, scale: float) -> None:
        """Cambia `Time.timeScale` en Unity (requiere `UrcVideoRecorder.cs` en la escena)."""
        self._recording_channel.set_time_scale(scale)

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

    @staticmethod
    def _flatten_observation(sensor_observations: list[np.ndarray]) -> np.ndarray:
        """Concatena las observaciones de todos los sensores en un único
        vector plano (ver docstring de la clase: solo importa preservar la
        estructura por sensor si hubiera una política con CNN, fuera de
        alcance por ahora)."""
        return np.concatenate([np.asarray(obs).reshape(-1) for obs in sensor_observations])
