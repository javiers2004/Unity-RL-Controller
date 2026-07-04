"""Prueba de extremo a extremo del currículo declarado en config: un entorno
con `bridge_options` (host/port de un servidor TCP de juguete) y `curriculum`
(dos lessons), verificando que `urc train` aplica de verdad los parámetros de
cada lesson según avanza la recompensa — el mismo flujo que se comprobó a
mano contra el servidor de juguete durante el desarrollo de esta fase.
"""

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


def _serve_curriculum_env(server_socket: socket.socket, received: list[dict]) -> None:
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
                done = step_count >= 2
                if done:
                    step_count = 0
                result = {"observation": [0.0], "reward": 1.0, "done": done, "info": {}}
            elif method == "observation_spec":
                result = {"shape": [1], "dtype": "float32"}
            elif method == "action_spec":
                result = {"shape": [1], "dtype": "float32", "discrete": False}
            elif method == "set_parameters":
                received.append(request.get("params", {}).get("parameters"))
                result = None
            else:
                writer.write(json.dumps({"error": f"método desconocido: {method}"}) + "\n")
                writer.flush()
                continue

            writer.write(json.dumps({"result": result}) + "\n")
            writer.flush()


@pytest.fixture
def curriculum_env_server() -> Iterator[tuple[str, int, list[dict]]]:
    received: list[dict] = []
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(1)
    host, port = server_socket.getsockname()

    thread = threading.Thread(
        target=_serve_curriculum_env, args=(server_socket, received), daemon=True
    )
    thread.start()

    yield host, port, received

    server_socket.close()
    thread.join(timeout=5)


def test_train_applies_curriculum_parameters_end_to_end(
    tmp_path: Path, curriculum_env_server: tuple[str, int, list[dict]]
):
    host, port, received_parameters = curriculum_env_server

    project = tmp_path / "urc.yaml"
    project.write_text(
        "bridge: socket\n"
        "env: maze-v1\n"
        "environments:\n"
        "  maze-v1:\n"
        "    bridge_options:\n"
        f"      host: {host}\n"
        f"      port: {port}\n"
        "    curriculum:\n"
        "      - parameters:\n"
        "          difficulty: 0.1\n"
        "        min_reward: 0.5\n"
        "        min_episodes: 1\n"
        "      - parameters:\n"
        "          difficulty: 0.9\n"
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--project",
            str(project),
            "--set",
            "training.max_steps=8",
            "--set",
            "hyperparameters.n_steps=4",
            "--set",
            "hyperparameters.batch_size=2",
            "--set",
            "hyperparameters.n_epochs=1",
            "--set",
            f"output_dir={tmp_path / 'runs'}",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert {"difficulty": 0.1} in received_parameters
    assert {"difficulty": 0.9} in received_parameters
