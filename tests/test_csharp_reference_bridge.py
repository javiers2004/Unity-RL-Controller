"""Prueba de extremo a extremo del bridge de referencia en C#
(examples/csharp_bridge/Program.cs): lo compila con el compilador que trae
.NET Framework en Windows (csc.exe, sin necesitar el SDK de .NET) y lo conecta
con un `ExternalProcessBridge` real, sin ningún mock de por medio.

Se salta limpiamente si no hay csc.exe disponible (p. ej. en el runner Linux
de CI) — es una validación real de interoperabilidad cuando se puede correr,
no un requisito para poder desarrollar el resto del proyecto.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from urc.bridges.external_bridge import ExternalProcessBridge

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "examples" / "csharp_bridge" / "Program.cs"

_CSC_CANDIDATES = (
    r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
    r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
)


def _find_csc() -> str | None:
    found = shutil.which("csc") or shutil.which("csc.exe")
    if found:
        return found
    for candidate in _CSC_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


CSC = _find_csc()

pytestmark = pytest.mark.skipif(
    CSC is None,
    reason="No se encontró csc.exe (compilador de .NET Framework, solo en Windows).",
)


@pytest.fixture(scope="module")
def compiled_bridge_exe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("csharp_bridge")
    exe_path = out_dir / "bridge.exe"
    result = subprocess.run(
        [CSC, "/nologo", "/reference:System.Web.Extensions.dll", f"/out:{exe_path}", str(SOURCE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"csc.exe falló:\n{result.stdout}\n{result.stderr}"
    return exe_path


def test_csharp_reference_bridge_round_trips_over_stdio(compiled_bridge_exe: Path):
    bridge = ExternalProcessBridge([str(compiled_bridge_exe)])
    try:
        obs = bridge.reset()
        assert obs == [0.0]

        obs_spec = bridge.observation_spec()
        assert obs_spec.shape == (1,)
        assert obs_spec.dtype == "float32"

        action_spec = bridge.action_spec()
        assert action_spec.shape == (1,)
        assert action_spec.discrete is False

        bridge.set_parameters({"difficulty": 0.5})  # no debe lanzar

        results = [bridge.step([0.1]) for _ in range(3)]
        assert [result.done for result in results] == [False, False, True]
        assert all(result.reward == 1.0 for result in results)
    finally:
        bridge.close()


def test_csharp_reference_bridge_reports_unknown_method_as_error(compiled_bridge_exe: Path):
    from urc.core.rpc import JsonLineRpcClient, RpcError

    rpc = JsonLineRpcClient.over_subprocess([str(compiled_bridge_exe)])
    try:
        with pytest.raises(RpcError):
            rpc.call("does_not_exist")
    finally:
        rpc.close()
