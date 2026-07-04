"""Tests de `urc visualize`.

`visualize()` en sí bloquea hasta Ctrl+C (por diseño, como el propio comando
`tensorboard`), así que no se invoca entero por CliRunner: se prueba
`_launch_tensorboard` (la parte real y testeable) contra logs reales, y la
propia CLI solo por su `--help` (que no bloquea) para comprobar que está bien
registrada.
"""

import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("tensorboard")

from urc.cli.main import app  # noqa: E402
from urc.cli.visualize import _launch_tensorboard  # noqa: E402

runner = CliRunner()


def test_launch_tensorboard_serves_a_working_dashboard(tmp_path: Path):
    url = _launch_tensorboard(tmp_path, port=0)

    assert url.startswith("http://")
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200


def test_visualize_help_is_registered_without_blocking():
    result = runner.invoke(app, ["visualize", "--help"])

    assert result.exit_code == 0
    assert "logdir" in result.stdout.lower() or "--port" in result.stdout
