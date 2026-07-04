"""Bridge de juguete usado solo por test_external_bridge.py.

Simula un plugin de bridge implementado en un proceso aparte (como estaría
escrito en cualquier otro lenguaje), hablando el protocolo JSON-RPC por stdio.
"""

import json
import sys


def main() -> None:
    for line in sys.stdin:
        request = json.loads(line)
        method = request["method"]

        if method == "reset":
            result: object = {"obs": [0.0, 0.0]}
        elif method == "step":
            result = {
                "observation": {"obs": [1.0, 1.0]},
                "reward": 1.0,
                "done": False,
                "info": {},
            }
        elif method == "observation_spec":
            result = {"shape": [2], "dtype": "float32"}
        elif method == "action_spec":
            result = {"shape": [1], "dtype": "float32", "discrete": False}
        else:
            print(json.dumps({"error": f"método desconocido: {method}"}), flush=True)
            continue

        print(json.dumps({"result": result}), flush=True)


if __name__ == "__main__":
    main()
