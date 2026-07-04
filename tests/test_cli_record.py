import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("stable_baselines3")

from urc.cli.main import app  # noqa: E402

runner = CliRunner()


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
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    return next((output_dir / "default").glob("*.zip"))


def test_record_writes_jsonl_trajectory_with_one_line_per_step(
    tmp_path: Path, toy_env_server: tuple[str, int]
):
    host, port = toy_env_server
    checkpoint = _train_checkpoint(tmp_path, host, port)

    result = runner.invoke(app, ["record", str(checkpoint), "--episodes", "2"])

    assert result.exit_code == 0, result.stdout + result.stderr
    output_path = checkpoint.with_name(f"replay_{checkpoint.stem}.jsonl")
    assert output_path.exists()

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    # El servidor de juguete usa episodios de longitud 2 -> 2 episodios x 2 pasos.
    assert len(lines) == 4
    assert {line["episode"] for line in lines} == {0, 1}
    assert lines[0]["done"] is False
    assert lines[1]["done"] is True
    assert all("observation" in line and "action" in line for line in lines)


def test_record_respects_custom_output_path(tmp_path: Path, toy_env_server: tuple[str, int]):
    host, port = toy_env_server
    checkpoint = _train_checkpoint(tmp_path, host, port)
    custom_output = tmp_path / "mi_replay.jsonl"

    result = runner.invoke(
        app, ["record", str(checkpoint), "--episodes", "1", "--output", str(custom_output)]
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert custom_output.exists()
