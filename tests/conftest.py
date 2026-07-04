"""Fixtures compartidas entre módulos de test."""

import json
import socket
import threading
from collections.abc import Iterator

import pytest


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
    """Servidor TCP de juguete que habla el protocolo JSON-RPC de `SocketBridge`:
    episodios de longitud 2, observación/acción continuas de tamaño 1. Sirve un
    único cliente y se cierra solo al final del test."""
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
