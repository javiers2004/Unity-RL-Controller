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
