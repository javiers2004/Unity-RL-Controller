from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_INFO_FILENAME = "run_info.json"


def write_run_info(
    checkpoint_dir: Path, *, bridge: str, bridge_options: dict[str, Any], algo: str, env: str
) -> None:
    """Guarda, junto a los checkpoints de un run, con qué bridge/algoritmo se
    entrenó. Así `urc eval`/`urc record` pueden cargar un checkpoint sin que
    el usuario tenga que repetir la misma config que usó `urc train`."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    info = {"bridge": bridge, "bridge_options": bridge_options, "algo": algo, "env": env}
    (checkpoint_dir / RUN_INFO_FILENAME).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_run_info(checkpoint: Path) -> dict[str, Any]:
    """Lee el `run_info.json` junto a `checkpoint`, si existe. Si no (p. ej. un
    checkpoint que no viene de `urc train`), devuelve un dict vacío: el
    llamador debe poder seguir funcionando con la config que le pasen a mano."""
    run_info_path = checkpoint.parent / RUN_INFO_FILENAME
    if not run_info_path.exists():
        return {}
    return json.loads(run_info_path.read_text(encoding="utf-8"))
