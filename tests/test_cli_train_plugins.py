"""Prueba de extremo a extremo del mecanismo de plugins de terceros: un
algoritmo definido por el usuario, en un .py suelto dentro de `./plugins/`
(sin publicarlo como paquete), se registra solo y `urc train` lo usa.

No necesita stable-baselines3: el algoritmo de juguete no entrena nada de
verdad, solo demuestra que el cableado (CLI -> plugins -> registry -> bridge
real) funciona para cualquier AlgorithmBackend de terceros.
"""

from pathlib import Path

from typer.testing import CliRunner

from urc.cli.main import app

runner = CliRunner()

_CUSTOM_ALGO_PLUGIN = '''\
from urc.core.contracts import AlgorithmBackend, Policy
from urc.core.registry import algorithms


class DummyPolicy(Policy):
    def predict(self, observation):
        return 0


@algorithms.register("dummy-custom")
class DummyAlgorithm(AlgorithmBackend):
    """Algoritmo de prueba: hace un reset y un step reales contra el bridge
    para demostrar que de verdad recibe uno funcional, y no entrena nada más."""

    def train(self, bridge, env_spec, config):
        bridge.reset()
        bridge.step(0)
        return DummyPolicy()

    def load(self, checkpoint_path):
        return DummyPolicy()
'''


def test_train_and_algo_list_pick_up_custom_plugin_from_plugins_folder(
    tmp_path: Path, monkeypatch, toy_env_server: tuple[str, int]
):
    # Un solo test cubre ambos comandos a propósito: `bridges`/`algorithms` son
    # registries globales del proceso, y dos tests registrando el mismo nombre
    # "dummy-custom" por separado chocarían entre sí (ValueError: ya registrado).
    host, port = toy_env_server
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "dummy_algo.py").write_text(_CUSTOM_ALGO_PLUGIN, encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    train_result = runner.invoke(
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
            "algo=dummy-custom",
        ],
    )

    assert train_result.exit_code == 0, train_result.stdout + train_result.stderr
    assert "Entrenamiento terminado." in train_result.stdout

    # Esto también ejercita que load_all_plugins() sea idempotente: es la
    # segunda vez que se cargan los plugins en el mismo proceso de test.
    list_result = runner.invoke(app, ["algo", "list"])

    assert list_result.exit_code == 0
    assert "dummy-custom" in list_result.stdout.splitlines()
