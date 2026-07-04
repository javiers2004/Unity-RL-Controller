from __future__ import annotations

from typing import Any

from urc.core.contracts import ActionSpec, BridgeAdapter, ObservationSpec, StepResult
from urc.core.registry import bridges
from urc.core.rpc import StdioRpcClient


@bridges.register("external")
class ExternalProcessBridge(BridgeAdapter):
    """Bridge que delega en un proceso externo, en cualquier lenguaje, vía JSON-RPC por stdio."""

    def __init__(self, command: list[str]) -> None:
        self._rpc = StdioRpcClient(command)

    def reset(self) -> Any:
        return self._rpc.call("reset")

    def step(self, action: Any) -> StepResult:
        result = self._rpc.call("step", {"action": action})
        return StepResult(
            observation=result["observation"],
            reward=result["reward"],
            done=result["done"],
            info=result.get("info", {}),
        )

    def observation_spec(self) -> ObservationSpec:
        spec = self._rpc.call("observation_spec")
        return ObservationSpec(shape=tuple(spec["shape"]), dtype=spec.get("dtype", "float32"))

    def action_spec(self) -> ActionSpec:
        spec = self._rpc.call("action_spec")
        return ActionSpec(
            shape=tuple(spec["shape"]),
            dtype=spec.get("dtype", "float32"),
            discrete=spec.get("discrete", False),
        )

    def close(self) -> None:
        self._rpc.close()
