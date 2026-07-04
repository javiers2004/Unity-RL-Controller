from __future__ import annotations

from typing import Any


def json_safe(value: Any) -> Any:
    """Convierte arrays/escalares de numpy (frecuentes al venir de SB3, de un
    bridge, o de cualquier librería del ecosistema Gym) a tipos nativos
    serializables en JSON, sin depender de numpy: cualquier objeto con
    `.tolist()` sirve."""
    to_list = getattr(value, "tolist", None)
    return to_list() if callable(to_list) else value
