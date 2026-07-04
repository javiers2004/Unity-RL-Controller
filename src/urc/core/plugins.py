from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

ENTRY_POINT_GROUPS = ("urc.bridges", "urc.algorithms", "urc.environments")


def load_entry_point_plugins() -> None:
    """Importa los plugins de terceros instalados como entry points de `urc`.

    Cada plugin se registra a sí mismo al ser importado (vía el decorador
    `@registry.register(...)` de `urc.core.registry`), así que basta con cargarlo.
    """
    for group in ENTRY_POINT_GROUPS:
        for entry_point in importlib.metadata.entry_points(group=group):
            entry_point.load()


def load_plugins_from_dir(directory: str | Path) -> None:
    """Importa cada archivo .py de `directory` para que se auto-registre.

    Permite añadir un plugin sin publicarlo como paquete instalable: basta con
    dejar un .py en la carpeta de plugins del proyecto (p. ej. `./plugins/`).
    """
    directory = Path(directory)
    if not directory.is_dir():
        return

    for path in sorted(directory.glob("*.py")):
        module_name = f"urc._plugins.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
