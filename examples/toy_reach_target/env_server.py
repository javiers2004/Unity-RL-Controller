"""Entorno de juguete "reach the target": un agente en una línea 1D tiene que
moverse hacia un punto objetivo. Habla el mismo protocolo JSON-RPC por líneas
que cualquier bridge de `urc` (ver PROTOCOL.md) sobre un socket TCP — no
depende de `urc` para nada, es un servidor TCP normal y corriente.

No hace falta Unity para probar `urc train` con esto: es el ejemplo más
sencillo posible para ver el pipeline completo funcionando de extremo a
extremo (bridge -> entrenamiento -> evaluación).

Uso:
    python env_server.py --port 9000
    # en otra terminal: urc train --set bridge_options.port=9000 --project urc.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import socket

STEP_SIZE = 0.1
SUCCESS_THRESHOLD = 0.05
MAX_STEPS_PER_EPISODE = 50
SUCCESS_BONUS = 10.0


class ReachTargetEpisode:
    def __init__(self) -> None:
        self.position = 0.0
        self.target = random.choice([-0.8, 0.8])
        self.steps = 0

    def reset(self) -> list[float]:
        self.position = 0.0
        self.target = random.choice([-0.8, 0.8])
        self.steps = 0
        return [self.target - self.position]

    def step(self, action: float) -> dict:
        self.position += max(-1.0, min(1.0, action)) * STEP_SIZE
        self.steps += 1
        distance = self.target - self.position

        reached = abs(distance) < SUCCESS_THRESHOLD
        timed_out = self.steps >= MAX_STEPS_PER_EPISODE
        done = reached or timed_out

        reward = -abs(distance)
        if reached:
            reward += SUCCESS_BONUS

        return {
            "observation": [distance],
            "reward": reward,
            "done": done,
            "info": {"success": reached},
        }


def serve(port: int) -> None:
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    print(f"reach-target escuchando en 127.0.0.1:{port}", flush=True)

    episode = ReachTargetEpisode()
    while True:
        conn, _ = server_socket.accept()
        with conn:
            reader = conn.makefile("r", encoding="utf-8", newline="\n")
            writer = conn.makefile("w", encoding="utf-8", newline="\n")
            for line in reader:
                request = json.loads(line)
                method = request["method"]

                if method == "reset":
                    result: object = episode.reset()
                elif method == "step":
                    action = request["params"]["action"][0]
                    result = episode.step(action)
                elif method == "observation_spec":
                    result = {"shape": [1], "dtype": "float32"}
                elif method == "action_spec":
                    result = {"shape": [1], "dtype": "float32", "discrete": False}
                elif method == "set_parameters":
                    result = None
                else:
                    writer.write(json.dumps({"error": f"método desconocido: {method}"}) + "\n")
                    writer.flush()
                    continue

                writer.write(json.dumps({"result": result}) + "\n")
                writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    serve(args.port)
