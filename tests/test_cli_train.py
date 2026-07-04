from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("stable_baselines3")

from urc.cli.main import app  # noqa: E402

runner = CliRunner()


def test_train_reports_config_error_and_exits_nonzero(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("logging:\n  backend: not-a-real-backend\n")

    result = runner.invoke(app, ["train", "--project", str(project)])

    assert result.exit_code == 1
    assert "logging.backend" in result.stderr


def test_train_reports_unknown_bridge_and_exits_nonzero(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("bridge: does-not-exist\n")

    result = runner.invoke(app, ["train", "--project", str(project)])

    assert result.exit_code == 1
    assert "does-not-exist" in result.stderr


def test_train_reports_unknown_algo_and_exits_nonzero(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("algo: does-not-exist\n")

    result = runner.invoke(app, ["train", "--project", str(project)])

    assert result.exit_code == 1
    assert "does-not-exist" in result.stderr


def test_train_end_to_end_against_socket_bridge(
    tmp_path: Path, toy_env_server: tuple[str, int]
):
    host, port = toy_env_server
    output_dir = tmp_path / "runs"

    result = runner.invoke(
        app,
        [
            "train",
            "--set",
            "bridge=socket",
            "--set",
            f"bridge_options.host={host}",
            "--set",
            f"bridge_options.port={port}",
            "--set",
            "training.max_steps=16",
            "--set",
            "training.checkpoint_every=8",
            "--set",
            "hyperparameters.n_steps=8",
            "--set",
            "hyperparameters.batch_size=4",
            "--set",
            "hyperparameters.n_epochs=1",
            "--set",
            f"output_dir={output_dir}",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Entrenamiento terminado." in result.stdout
    checkpoints = list((output_dir / "default").glob("*.zip"))
    assert len(checkpoints) >= 1
