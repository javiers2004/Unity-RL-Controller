from urc.bridges.external_bridge import ExternalProcessBridge
from urc.bridges.socket_bridge import SocketBridge
from urc.core.registry import bridges

__all__ = ["ExternalProcessBridge", "SocketBridge"]

# `mlagents_bridge` no se importa aquí a propósito: depende del extra opcional
# "mlagents" (pesado: grpc, protobuf, numpy...). Se registra como disponible
# bajo demanda: `bridges.get("mlagents")` lo importa la primera vez que se pide
# de verdad (p. ej. desde `urc train`), sin forzar la dependencia a quien no la use.
bridges.register_lazy(
    "mlagents", "urc.bridges.mlagents_bridge", install_hint='pip install "urc[mlagents]"'
)
