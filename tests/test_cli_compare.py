import json
from pathlib import Path

from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()


def _write_eval_result(path: Path, *, mean_reward: float, success_rate: float | None) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": str(path),
                "episodes": [{"reward": mean_reward, "length": 2}],
                "mean_reward": mean_reward,
                "std_reward": 0.0,
                "mean_length": 2.0,
                "success_rate": success_rate,
            }
        ),
        encoding="utf-8",
    )


def test_compare_reads_eval_json_files_directly(tmp_path: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_eval_result(a, mean_reward=1.0, success_rate=0.5)
    _write_eval_result(b, mean_reward=2.0, success_rate=None)

    result = runner.invoke(app, ["compare", str(a), str(b)])

    assert result.exit_code == 0
    assert "1.000" in result.stdout
    assert "2.000" in result.stdout
    assert "50.0%" in result.stdout
    assert "N/A" in result.stdout


def test_compare_resolves_checkpoint_to_sibling_eval_file(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint_100_steps.zip"
    checkpoint.write_bytes(b"")
    _write_eval_result(
        tmp_path / "eval_checkpoint_100_steps.json", mean_reward=3.0, success_rate=1.0
    )

    result = runner.invoke(app, ["compare", str(checkpoint)])

    assert result.exit_code == 0
    assert "3.000" in result.stdout


def test_compare_resolves_directory_to_most_recent_eval_file(tmp_path: Path):
    import time

    old = tmp_path / "eval_old.json"
    _write_eval_result(old, mean_reward=1.0, success_rate=None)
    time.sleep(0.01)
    new = tmp_path / "eval_new.json"
    _write_eval_result(new, mean_reward=9.0, success_rate=None)

    result = runner.invoke(app, ["compare", str(tmp_path)])

    assert result.exit_code == 0
    assert "9.000" in result.stdout
    assert "1.000" not in result.stdout


def test_compare_missing_eval_result_exits_nonzero(tmp_path: Path):
    checkpoint = tmp_path / "no_eval_yet.zip"
    checkpoint.write_bytes(b"")

    result = runner.invoke(app, ["compare", str(checkpoint)])

    assert result.exit_code == 1
    assert "urc eval" in result.stderr
