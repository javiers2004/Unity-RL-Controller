from pathlib import Path

from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()


def test_init_creates_project_structure(tmp_path: Path):
    project_dir = tmp_path / "mi-proyecto"

    result = runner.invoke(app, ["init", str(project_dir)])

    assert result.exit_code == 0
    assert (project_dir / "urc.yaml").exists()
    assert (project_dir / "experiments").is_dir()
    assert "bridge: mlagents" in (project_dir / "urc.yaml").read_text(encoding="utf-8")


def test_init_refuses_to_overwrite_non_empty_directory(tmp_path: Path):
    project_dir = tmp_path / "mi-proyecto"
    project_dir.mkdir()
    (project_dir / "algo.txt").write_text("ya había algo aquí")

    result = runner.invoke(app, ["init", str(project_dir)])

    assert result.exit_code == 1
    assert "no está vacío" in result.stderr


def test_init_succeeds_on_existing_empty_directory(tmp_path: Path):
    project_dir = tmp_path / "mi-proyecto"
    project_dir.mkdir()

    result = runner.invoke(app, ["init", str(project_dir)])

    assert result.exit_code == 0
    assert (project_dir / "urc.yaml").exists()
