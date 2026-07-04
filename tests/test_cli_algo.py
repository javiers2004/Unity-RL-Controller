import pytest
from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()


def test_algo_list_shows_built_in_names_without_needing_them_installed():
    # `names()` no importa nada: los backends con dependencias opcionales
    # (sb3-ppo, sb3-sac) deben aparecer en la lista aunque no estén instalados.
    result = runner.invoke(app, ["algo", "list"])

    assert result.exit_code == 0
    names = result.stdout.splitlines()
    assert "sb3-ppo" in names
    assert "sb3-sac" in names


def test_algo_info_reports_unknown_algorithm_and_exits_nonzero():
    result = runner.invoke(app, ["algo", "info", "does-not-exist"])

    assert result.exit_code == 1
    assert "does-not-exist" in result.stderr


def test_algo_info_prints_docstring_for_a_real_backend():
    pytest.importorskip("stable_baselines3")

    result = runner.invoke(app, ["algo", "info", "sb3-ppo"])

    assert result.exit_code == 0
    assert "sb3-ppo" in result.stdout
    assert "PPO" in result.stdout
