from typer.testing import CliRunner

from urc import __version__
from urc.cli.main import app

runner = CliRunner()


def test_version_command_prints_installed_version():
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout
