from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("mlagents_envs")

from urc.cli.main import app  # noqa: E402
from urc.core.contracts import ActionSpec, ObservationSpec  # noqa: E402

runner = CliRunner()


class FakeBridge:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False

    def reset(self):
        return None

    def observation_spec(self):
        return ObservationSpec(shape=(2,))

    def action_spec(self):
        return ActionSpec(shape=(1,), discrete=False)

    def close(self):
        self.closed = True


def test_env_launch_reports_connection_and_specs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("urc.bridges.mlagents_bridge.MLAgentsBridge", FakeBridge)

    result = runner.invoke(app, ["env", "launch"])

    assert result.exit_code == 0
    assert "Conexión establecida correctamente." in result.stdout
    assert "shape=(2,)" in result.stdout


def test_env_launch_with_env_uses_its_declared_build_path_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured_kwargs: dict[str, object] = {}

    class CapturingFakeBridge(FakeBridge):
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr("urc.bridges.mlagents_bridge.MLAgentsBridge", CapturingFakeBridge)

    project = tmp_path / "urc.yaml"
    project.write_text(
        "environments:\n"
        "  maze-v1:\n"
        "    build_path: builds/maze.exe\n"
        "    bridge_options:\n"
        "      no_graphics: true\n"
    )

    result = runner.invoke(
        app, ["env", "launch", "--env", "maze-v1", "--project", str(project)]
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert captured_kwargs == {"no_graphics": True, "file_name": "builds/maze.exe"}


def test_env_launch_explicit_executable_overrides_environment_build_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured_kwargs: dict[str, object] = {}

    class CapturingFakeBridge(FakeBridge):
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr("urc.bridges.mlagents_bridge.MLAgentsBridge", CapturingFakeBridge)

    project = tmp_path / "urc.yaml"
    project.write_text("environments:\n  maze-v1:\n    build_path: builds/maze.exe\n")

    result = runner.invoke(
        app,
        [
            "env",
            "launch",
            "--env",
            "maze-v1",
            "--executable",
            "otro-build.exe",
            "--project",
            str(project),
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert captured_kwargs["file_name"] == "otro-build.exe"


def test_env_launch_with_unknown_env_exits_nonzero(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("bridge: mlagents\n")

    result = runner.invoke(
        app, ["env", "launch", "--env", "does-not-exist", "--project", str(project)]
    )

    assert result.exit_code == 1
    assert "does-not-exist" in result.stderr
