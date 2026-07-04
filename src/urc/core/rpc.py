from __future__ import annotations

import json
import subprocess
from typing import Any


class ExternalProcessError(RuntimeError):
    """El proceso externo devolvió un error o no respondió el protocolo esperado."""


class StdioRpcClient:
    """Protocolo JSON-RPC minimalista por stdin/stdout con un proceso externo.

    Cada llamada envía una línea `{"method": ..., "params": ...}` y espera de
    vuelta una línea `{"result": ...}` o `{"error": ...}`. No depende de nada
    más que JSON, así que cualquier lenguaje puede implementar el otro extremo.
    """

    def __init__(self, command: list[str]) -> None:
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def call(self, method: str, params: Any = None) -> Any:
        if self._process.stdin is None or self._process.stdout is None:
            raise ExternalProcessError("El proceso externo no tiene stdio conectado")

        self._process.stdin.write(json.dumps({"method": method, "params": params}) + "\n")
        self._process.stdin.flush()

        line = self._process.stdout.readline()
        if not line:
            raise ExternalProcessError(f"El proceso externo se cerró sin responder a '{method}'")

        response = json.loads(line)
        if "error" in response:
            raise ExternalProcessError(str(response["error"]))
        return response.get("result")

    def close(self) -> None:
        if self._process.stdin:
            self._process.stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)
