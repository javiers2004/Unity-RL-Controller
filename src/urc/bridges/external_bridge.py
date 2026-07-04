from __future__ import annotations

from urc.core.registry import bridges
from urc.core.rpc import JsonLineBridge, JsonLineRpcClient


@bridges.register("external")
class ExternalProcessBridge(JsonLineBridge):
    """Bridge que delega en un proceso externo, en cualquier lenguaje, vía JSON-RPC por stdio."""

    def __init__(self, command: list[str]) -> None:
        super().__init__(JsonLineRpcClient.over_subprocess(command))
