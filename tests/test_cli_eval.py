from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("stable_baselines3")

from urc.cli.main import app  # noqa: E402

runner = CliRunner()

_TRAIN_ARGS = [
    "--set",
    "training.max_steps=8",
    "--set",
    "training.checkpoint_every=8",
    "--set",
    "hyperparameters.n_steps=4",
    "--set",
    "hyperparameters.batch_size=2",
    "--set",
    "hyperparameters.n_epochs=1",
]


def _train_checkpoint(tmp_path: Path, host: str, port: int) -> Path:
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
            f"output_dir={output_dir}",
            *_TRAIN_ARGS,
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    checkpoints = list((output_dir / "default").glob("*.zip"))
    assert len(checkpoints) == 1
    return checkpoints[0]


def test_eval_uses_run_info_to_reconnect_without_repeating_config(
    tmp_path: Path, toy_env_server: tuple[str, int]
):
    host, port = toy_env_server
    checkpoint = _train_checkpoint(tmp_path, host, port)

    result = runner.invoke(app, ["eval", str(checkpoint), "--episodes", "2"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Episodios:        2" in result.stdout
    assert "Reward medio:" in result.stdout

    output_path = checkpoint.with_name(f"eval_{checkpoint.stem}.json")
    assert output_path.exists()


def test_eval_reports_na_success_rate_without_info_or_threshold(
    tmp_path: Path, toy_env_server: tuple[str, int]
):
    host, port = toy_env_server
    checkpoint = _train_checkpoint(tmp_path, host, port)

    result = runner.invoke(app, ["eval", str(checkpoint), "--episodes", "1"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "N/A" in result.stdout


def test_eval_with_success_threshold_computes_success_rate(
    tmp_path: Path, toy_env_server: tuple[str, int]
):
    host, port = toy_env_server
    checkpoint = _train_checkpoint(tmp_path, host, port)

    result = runner.invoke(
        app,
        ["eval", str(checkpoint), "--episodes", "2", "--success-threshold", "0.5"],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Tasa de éxito:    100.0%" in result.stdout


def test_eval_unknown_checkpoint_bridge_can_be_overridden_with_set(tmp_path: Path):
    checkpoint = tmp_path / "orphan.zip"
    checkpoint.write_bytes(b"")

    result = runner.invoke(app, ["eval", str(checkpoint), "--set", "bridge=does-not-exist"])

    assert result.exit_code == 1
    assert "does-not-exist" in result.stderr
