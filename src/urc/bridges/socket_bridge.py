from __future__ import annotations

from urc.core.registry import bridges
from urc.core.rpc import JsonLineBridge, JsonLineRpcClient


@bridges.register("socket")
class SocketBridge(JsonLineBridge):
    """Bridge que se conecta a un entorno remoto por TCP con el mismo protocolo
    JSON-RPC por líneas que `ExternalProcessBridge`. Sirve para demostrar que el
    contrato `BridgeAdapter` es intercambiable también con un transporte de red,
    no solo con subprocesos o con el gRPC interno de ML-Agents."""

    def __init__(self, host: str, port: int) -> None:
        super().__init__(JsonLineRpcClient.over_socket(host, port))
