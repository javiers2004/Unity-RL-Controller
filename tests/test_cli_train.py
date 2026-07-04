import json
import socket
import threading
from collections.abc import Iterator
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


def _serve_fixed_length_episodes(server_socket: socket.socket, episode_length: int = 2) -> None:
    conn, _ = server_socket.accept()
    with conn:
        reader = conn.makefile("r", encoding="utf-8", newline="\n")
        writer = conn.makefile("w", encoding="utf-8", newline="\n")
        step_count = 0
        for line in reader:
            request = json.loads(line)
            method = request["method"]

            if method == "reset":
                step_count = 0
                result: object = [0.0]
            elif method == "step":
                step_count += 1
                done = step_count >= episode_length
                if done:
                    step_count = 0
                result = {"observation": [0.0], "reward": 1.0, "done": done, "info": {}}
            elif method == "observation_spec":
                result = {"shape": [1], "dtype": "float32"}
            elif method == "action_spec":
                result = {"shape": [1], "dtype": "float32", "discrete": False}
            else:
                writer.write(json.dumps({"error": f"método desconocido: {method}"}) + "\n")
                writer.flush()
                continue

            writer.write(json.dumps({"result": result}) + "\n")
            writer.flush()


@pytest.fixture
def toy_env_server() -> Iterator[tuple[str, int]]:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(1)
    host, port = server_socket.getsockname()

    thread = threading.Thread(
        target=_serve_fixed_length_episodes, args=(server_socket,), daemon=True
    )
    thread.start()

    yield host, port

    server_socket.close()
    thread.join(timeout=5)


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
