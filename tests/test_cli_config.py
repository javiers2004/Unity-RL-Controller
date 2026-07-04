from pathlib import Path

from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()


def test_config_show_prints_resolved_yaml(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("bridge: socket\n")

    result = runner.invoke(app, ["config", "show", "--project", str(project)])

    assert result.exit_code == 0
    assert "bridge: socket" in result.stdout
    assert "algo: sb3-ppo" in result.stdout


def test_config_show_applies_set_overrides(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("bridge: mlagents\n")

    result = runner.invoke(
        app,
        ["config", "show", "--project", str(project), "--set", "bridge=socket"],
    )

    assert result.exit_code == 0
    assert "bridge: socket" in result.stdout


def test_config_validate_reports_success(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("bridge: socket\n")

    result = runner.invoke(app, ["config", "validate", "--project", str(project)])

    assert result.exit_code == 0
    assert "válida" in result.stdout


def test_config_validate_reports_readable_error_and_exits_nonzero(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("logging:\n  backend: not-a-real-backend\n")

    result = runner.invoke(app, ["config", "validate", "--project", str(project)])

    assert result.exit_code == 1
    assert "logging.backend" in result.stderr


def test_config_diff_reports_changed_fields(tmp_path: Path):
    a = tmp_path / "a.yaml"
    a.write_text("bridge: mlagents\n")
    b = tmp_path / "b.yaml"
    b.write_text("bridge: socket\n")

    result = runner.invoke(app, ["config", "diff", str(a), str(b)])

    assert result.exit_code == 0
    assert "bridge: 'mlagents' -> 'socket'" in result.stdout


def test_config_diff_reports_no_differences(tmp_path: Path):
    a = tmp_path / "a.yaml"
    a.write_text("bridge: mlagents\n")
    b = tmp_path / "b.yaml"
    b.write_text("bridge: mlagents\n")

    result = runner.invoke(app, ["config", "diff", str(a), str(b)])

    assert result.exit_code == 0
    assert "Sin diferencias." in result.stdout
