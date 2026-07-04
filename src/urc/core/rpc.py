from __future__ import annotations

import json
import socket
import subprocess
from collections.abc import Callable
from typing import IO, Any

from urc.core.contracts import ActionSpec, BridgeAdapter, ObservationSpec, StepResult


class RpcError(RuntimeError):
    """El extremo remoto devolvió un error o no respondió el protocolo esperado."""


def _json_safe(value: Any) -> Any:
    """Convierte arrays/escalares de numpy (frecuentes al venir de SB3 u otras
    librerías del ecosistema Gym) a tipos nativos serializables en JSON, sin
    depender de numpy: cualquier objeto con `.tolist()` sirve."""
    to_list = getattr(value, "tolist", None)
    return to_list() if callable(to_list) else value


class JsonLineRpcClient:
    """Protocolo JSON-RPC minimalista, una petición/respuesta por línea de texto.

    No depende de nada más que JSON, así que cualquier lenguaje puede
    implementar el otro extremo, ya sea un subproceso (stdio) o un servicio
    de red (socket TCP): basta con leer/escribir líneas de texto.
    """

    def __init__(self, reader: IO[str], writer: IO[str], *, on_close: Callable[[], None]) -> None:
        self._reader = reader
        self._writer = writer
        self._on_close = on_close

    @classmethod
    def over_subprocess(cls, command: list[str]) -> JsonLineRpcClient:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            raise RpcError("No se pudo conectar stdin/stdout del proceso externo")

        def on_close() -> None:
            process.stdin.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

        return cls(process.stdout, process.stdin, on_close=on_close)

    @classmethod
    def over_socket(cls, host: str, port: int, *, timeout: float | None = 10) -> JsonLineRpcClient:
        sock = socket.create_connection((host, port), timeout=timeout)
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")

        def on_close() -> None:
            reader.close()
            writer.close()
            sock.close()

        return cls(reader, writer, on_close=on_close)

    def call(self, method: str, params: Any = None) -> Any:
        self._writer.write(json.dumps({"method": method, "params": params}) + "\n")
        self._writer.flush()

        line = self._reader.readline()
        if not line:
            raise RpcError(f"El extremo remoto se cerró sin responder a '{method}'")

        response = json.loads(line)
        if "error" in response:
            raise RpcError(str(response["error"]))
        return response.get("result")

    def close(self) -> None:
        self._on_close()


class JsonLineBridge(BridgeAdapter):
    """Bridge genérico sobre `JsonLineRpcClient`: mapea el contrato 1:1 al protocolo.

    Sirve de base para cualquier bridge que hable JSON por líneas, sea cual
    sea el transporte (subproceso, socket...). Los transportes concretos solo
    necesitan construir el `JsonLineRpcClient` adecuado.
    """

    def __init__(self, rpc: JsonLineRpcClient) -> None:
        self._rpc = rpc

    def reset(self) -> Any:
        return self._rpc.call("reset")

    def step(self, action: Any) -> StepResult:
        result = self._rpc.call("step", {"action": _json_safe(action)})
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
        discrete_branches = spec.get("discrete_branches")
        return ActionSpec(
            shape=tuple(spec["shape"]),
            dtype=spec.get("dtype", "float32"),
            discrete=spec.get("discrete", False),
            discrete_branches=tuple(discrete_branches) if discrete_branches else None,
        )

    def close(self) -> None:
        self._rpc.close()
