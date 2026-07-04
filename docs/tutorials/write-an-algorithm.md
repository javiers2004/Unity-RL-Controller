# Escribir un algoritmo propio

A diferencia de los bridges, los algoritmos son **plugins de Python**: implementan la interfaz
`AlgorithmBackend` (`urc.core.contracts`). No hace falta tocar el código de `urc` ni publicar un
paquete — basta con un `.py` en una carpeta `plugins/` junto a tu `urc.yaml`.

## El contrato

```python
from urc.core.contracts import AlgorithmBackend, Policy


class MiPolitica(Policy):
    def predict(self, observation):
        ...  # devuelve la acción para esa observación


class MiAlgoritmo(AlgorithmBackend):
    def train(self, bridge, env_spec, config):
        # bridge: BridgeAdapter ya conectado (reset()/step()/...)
        # env_spec: metadatos del entorno (nombre, curriculum, parameters...)
        # config: dict con "hyperparameters", "training", "checkpoint_dir", "resume_from"...
        ...
        return MiPolitica()

    def load(self, checkpoint_path):
        ...
        return MiPolitica()
```

## Registrarlo

En `plugins/mi_algoritmo.py`, dentro de tu proyecto:

```python
from urc.core.contracts import AlgorithmBackend, Policy
from urc.core.registry import algorithms


class MiPolitica(Policy):
    def predict(self, observation):
        return 0  # ejemplo: acción fija


@algorithms.register("mi-algoritmo")
class MiAlgoritmo(AlgorithmBackend):
    def train(self, bridge, env_spec, config):
        bridge.reset()
        # ... tu lógica de entrenamiento real aquí ...
        return MiPolitica()

    def load(self, checkpoint_path):
        return MiPolitica()
```

Y a usarlo, sin instalar nada más:

```bash
urc train --set algo=mi-algoritmo
urc algo list       # aparece "mi-algoritmo" en la lista
urc algo info mi-algoritmo
```

`urc train`/`urc algo list`/`urc algo info` cargan automáticamente cualquier `.py` en `./plugins/`
antes de resolver el algoritmo por nombre (ver Fase 6 del [Roadmap](../roadmap.md)).

## Reutilizar la mecánica de Stable-Baselines3

Si tu algoritmo también es de SB3 (una tercera opción además de PPO/SAC), no hace falta
reimplementar entrenar/reanudar/checkpoint desde cero — hereda de la base compartida:

```python
from stable_baselines3 import A2C

from urc.algorithms.sb3_base import SB3Backend
from urc.core.registry import algorithms


@algorithms.register("sb3-a2c")
class SB3A2CBackend(SB3Backend):
    algorithm_cls = A2C
```

Así obtienes checkpointing, `--resume`, TensorBoard/wandb y soporte de currículo gratis (ver
`urc.algorithms.sb3_ppo`/`sb3_sac` para ejemplos reales de este mismo patrón).

## Publicarlo como paquete instalable (opcional)

Si quieres distribuir tu algoritmo como un paquete pip en vez de un `.py` suelto, decláralo como
entry point en tu `pyproject.toml`:

```toml
[project.entry-points."urc.algorithms"]
mi_algoritmo = "mi_paquete.algoritmo:MiAlgoritmo"
```

`urc` lo descubre igual que los de `plugins/`, sin ninguna diferencia desde `urc train`.
