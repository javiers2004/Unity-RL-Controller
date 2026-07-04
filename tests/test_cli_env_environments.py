"""Tests de `urc env list/describe/create`: no necesitan mlagents_envs, son
solo config + registry (a diferencia de `urc env launch`, ver test_cli_env.py).
"""

from pathlib import Path

from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()


def _project_with_environment(tmp_path: Path) -> Path:
    project = tmp_path / "urc.yaml"
    project.write_text(
        "environments:\n"
        "  maze-v1:\n"
        "    build_path: builds/maze.exe\n"
        "    bridge_options:\n"
        "      no_graphics: true\n"
        "    parameters:\n"
        "      difficulty: 0.5\n"
    )
    return project


def test_env_list_shows_declared_environments(tmp_path: Path):
    # No se compara la lista completa: `environments` es un registry global
    # del proceso, así que otros tests que hayan declarado otros entornos en
    # la misma sesión también aparecerán — igual que en producción cada
    # comando `urc` es un proceso nuevo, así que esto no pasa nunca de verdad.
    project = _project_with_environment(tmp_path)

    result = runner.invoke(app, ["env", "list", "--project", str(project)])

    assert result.exit_code == 0
    assert "maze-v1" in result.stdout.splitlines()


def test_env_describe_prints_environment_details(tmp_path: Path):
    project = _project_with_environment(tmp_path)

    result = runner.invoke(app, ["env", "describe", "maze-v1", "--project", str(project)])

    assert result.exit_code == 0
    assert "build_path: builds/maze.exe" in result.stdout
    assert "no_graphics" in result.stdout
    assert "difficulty" in result.stdout


def test_env_describe_unknown_environment_exits_nonzero(tmp_path: Path):
    project = _project_with_environment(tmp_path)

    result = runner.invoke(app, ["env", "describe", "does-not-exist", "--project", str(project)])

    assert result.exit_code == 1
    assert "does-not-exist" in result.stderr


def test_env_create_adds_new_environment_to_project_yaml(tmp_path: Path):
    project = tmp_path / "urc.yaml"

    result = runner.invoke(
        app,
        [
            "env",
            "create",
            "maze-v2",
            "--build-path",
            "builds/maze2.exe",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 0
    assert project.exists()

    list_result = runner.invoke(app, ["env", "list", "--project", str(project)])
    assert "maze-v2" in list_result.stdout.splitlines()


def test_env_create_refuses_to_overwrite_existing_environment(tmp_path: Path):
    project = _project_with_environment(tmp_path)

    result = runner.invoke(app, ["env", "create", "maze-v1", "--project", str(project)])

    assert result.exit_code == 1
    assert "ya existe" in result.stderr
