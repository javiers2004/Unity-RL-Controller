from urc.bridges.external_bridge import ExternalProcessBridge
from urc.bridges.socket_bridge import SocketBridge

__all__ = ["ExternalProcessBridge", "SocketBridge"]

# `mlagents_bridge` no se importa aquí a propósito: depende del extra opcional
# "mlagents" (pesado: grpc, protobuf, numpy...). Se importa bajo demanda desde
# donde haga falta (p. ej. `urc env launch`), momento en que se registra como
# bridge "mlagents".
