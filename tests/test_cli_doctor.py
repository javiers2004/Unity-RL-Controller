import importlib

from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()

_real_import_module = importlib.import_module


def test_doctor_reports_missing_optional_dependency(monkeypatch):
    def fake_import_module(name: str):
        if name == "stable_baselines3":
            raise ImportError("simulado: no instalado")
        return _real_import_module(name)

    monkeypatch.setattr("urc.cli.doctor.importlib.import_module", fake_import_module)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "[ ]" in result.stdout
    assert 'pip install "urc[sb3]"' in result.stdout


def test_doctor_prints_python_and_urc_version():
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Python" in result.stdout
    assert "urc" in result.stdout
