from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

ENTRY_POINT_GROUPS = ("urc.bridges", "urc.algorithms", "urc.environments")


def _plugin_module_name(path: Path) -> str:
    """Nombre de módulo determinista y único por ruta absoluta.

    No basta con el nombre del archivo: dos proyectos distintos pueden tener
    cada uno su propio `plugins/mi_algo.py`, y deben poder convivir en el
    mismo proceso (p. ej. en los tests) sin pisarse en `sys.modules`.
    """
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    return f"urc._plugins.{digest}_{path.stem}"


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
    Idempotente: si el mismo archivo ya se cargó en este proceso (p. ej. porque
    `load_all_plugins` se llama una vez por comando y un test invoca varios
    comandos seguidos), no se vuelve a ejecutar ni a re-registrar.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return

    for path in sorted(directory.glob("*.py")):
        module_name = _plugin_module_name(path)
        if module_name in sys.modules:
            continue

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def load_all_plugins(plugins_dir: str | Path = "plugins") -> None:
    """Carga todos los plugins disponibles antes de resolver algo por nombre:
    entry points instalados + la carpeta de plugins local del proyecto
    (relativa al directorio desde el que se ejecuta `urc`)."""
    load_entry_point_plugins()
    load_plugins_from_dir(plugins_dir)
