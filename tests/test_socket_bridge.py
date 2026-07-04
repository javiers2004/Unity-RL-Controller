import json
import socket
import threading
from collections.abc import Iterator

import pytest

from urc.bridges.socket_bridge import SocketBridge


def _serve_one_client(server_socket: socket.socket) -> None:
    conn, _ = server_socket.accept()
    with conn:
        reader = conn.makefile("r", encoding="utf-8", newline="\n")
        writer = conn.makefile("w", encoding="utf-8", newline="\n")
        for line in reader:
            request = json.loads(line)
            method = request["method"]

            if method == "reset":
                result: object = {"obs": [0.0]}
            elif method == "step":
                received_action = request.get("params", {}).get("action")
                result = {
                    "observation": {"obs": [1.0]},
                    "reward": 2.0,
                    "done": True,
                    "info": {"received_action": received_action},
                }
            elif method == "observation_spec":
                result = {"shape": [1], "dtype": "float32"}
            elif method == "action_spec":
                result = {"shape": [1], "dtype": "float32", "discrete": True}
            elif method == "set_parameters":
                result = {"received_parameters": request.get("params", {}).get("parameters")}
            else:
                writer.write(json.dumps({"error": f"método desconocido: {method}"}) + "\n")
                writer.flush()
                continue

            writer.write(json.dumps({"result": result}) + "\n")
            writer.flush()


@pytest.fixture
def echo_server() -> Iterator[tuple[str, int]]:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 0))
    server_socket.listen(1)
    host, port = server_socket.getsockname()

    thread = threading.Thread(target=_serve_one_client, args=(server_socket,), daemon=True)
    thread.start()

    yield host, port

    server_socket.close()
    thread.join(timeout=5)


def test_socket_bridge_round_trips_over_tcp(echo_server: tuple[str, int]):
    host, port = echo_server

    bridge = SocketBridge(host, port)
    try:
        bridge.reset()

        obs_spec = bridge.observation_spec()
        assert obs_spec.shape == (1,)

        action_spec = bridge.action_spec()
        assert action_spec.discrete is True

        result = bridge.step(action=1)
        assert result.reward == 2.0
        assert result.done is True
        assert result.observation == {"obs": [1.0]}
    finally:
        bridge.close()


class _FakeNdarray:
    """Doble mínimo de numpy.ndarray: solo lo que _json_safe necesita (.tolist())."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def test_socket_bridge_serializes_array_like_actions(echo_server: tuple[str, int]):
    host, port = echo_server

    bridge = SocketBridge(host, port)
    try:
        bridge.reset()
        result = bridge.step(action=_FakeNdarray([0.25, -0.5]))
        assert result.info["received_action"] == [0.25, -0.5]
    finally:
        bridge.close()


def test_socket_bridge_set_parameters_sends_them_over_rpc(echo_server: tuple[str, int]):
    host, port = echo_server

    bridge = SocketBridge(host, port)
    try:
        bridge.reset()
        bridge.set_parameters({"difficulty": 0.5})
    finally:
        bridge.close()
