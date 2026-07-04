# Unity-RL-Controller (`urc`)

[![CI](https://github.com/javiers2004/Unity-RL-Controller/actions/workflows/ci.yml/badge.svg)](https://github.com/javiers2004/Unity-RL-Controller/actions/workflows/ci.yml)
[![Unity integration](https://github.com/javiers2004/Unity-RL-Controller/actions/workflows/unity-integration.yml/badge.svg)](https://github.com/javiers2004/Unity-RL-Controller/actions/workflows/unity-integration.yml)
[![Docs](https://github.com/javiers2004/Unity-RL-Controller/actions/workflows/docs.yml/badge.svg)](https://javiers2004.github.io/Unity-RL-Controller/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Librería + CLI para controlar entrenamientos de Reinforcement Learning en Unity desde la terminal: bridge Unity↔código, algoritmos, mapas, hiperparámetros, evaluación y visualización, todo con comandos simples y componentes intercambiables.

El diseño completo y el plan de desarrollo por fases están en [ROADMAP.md](ROADMAP.md). La
especificación del protocolo para escribir bridges en cualquier lenguaje está en
[PROTOCOL.md](PROTOCOL.md). El historial de cambios por fase está en [CHANGELOG.md](CHANGELOG.md).
Hay 3 proyectos de ejemplo completos en [`examples/`](examples/) (sin Unity, con un bridge en C#,
y con Unity real) y un sitio de documentación navegable en
[javiers2004.github.io/Unity-RL-Controller](https://javiers2004.github.io/Unity-RL-Controller/)
(o `mkdocs serve` para verlo en local). ¿Quieres contribuir? Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## Demo

Entrenamiento de extremo a extremo sin necesitar Unity instalado — entorno de juguete 1D, PPO,
[`examples/toy_reach_target/`](examples/toy_reach_target) (transcripción real, sin editar):

```console
$ urc train
Bridge: socket  Algoritmo: sb3-ppo  Entorno: default
Checkpoints en: runs\default
Entrenamiento terminado.

$ urc eval runs/default/checkpoint_20000_steps.zip --episodes 30
Episodios:        30
Reward medio:     7.153 ± 0.009
Duración media:   8.0 pasos
Tasa de éxito:    100.0%
Resultados guardados en: runs\default\eval_checkpoint_20000_steps.json
```

El mismo flujo funciona igual contra Unity real cambiando solo la config (`bridge: mlagents` en
vez de `bridge: socket`) — ver [`examples/unity_basic_ppo/`](examples/unity_basic_ppo). No hay
todavía un GIF del agente entrenando dentro del editor de Unity: es una grabación de pantalla, no
algo que se pueda generar aquí — si quieres aportar una, ver
["Grabar una demo visual"](CONTRIBUTING.md#grabar-una-demo-visual) en CONTRIBUTING.md.

> **Estado actual**: Fases 1-12 completadas (empaquetado, sitio de documentación, 3 ejemplos
> end-to-end, CI con un build headless real de Unity, y guías de contribución/código de conducta,
> todo verificado — ver el ROADMAP). Resto del recorrido: esqueleto, contratos/plugins,
> bridge de ML-Agents verificado contra Unity real, configuración jerárquica, `urc train` de
> extremo a extremo (PPO o SAC de Stable-Baselines3 sobre cualquier bridge, checkpointing y
> `--resume`), algoritmos de terceros vía `./plugins/`, entornos declarados en config con
> currículo/domain randomization reales, `urc eval/compare/record`, TensorBoard/wandb +
> `urc visualize`, y un protocolo out-of-process verificado contra un bridge de referencia en C#.

## Instalación (desarrollo)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (git-bash) — usa .venv\Scripts\activate en cmd/PowerShell
pip install -e ".[dev]"
```

Para usar el bridge por defecto contra Unity (ML-Agents), instala también el extra `mlagents`
(requiere tener Unity + un proyecto con el paquete `com.unity.ml-agents` para probarlo de verdad).
Para entrenar de verdad hace falta además el extra `sb3` (instala PyTorch, pesado — varios cientos
de MB). `all` instala ambos:

```bash
pip install -e ".[dev,all]"       # todo (recomendado para desarrollo)
pip install -e ".[dev,mlagents]"  # solo el bridge de Unity
pip install -e ".[dev,sb3]"       # entrenamiento + TensorBoard + barra de progreso
pip install -e ".[dev,wandb]"     # + Weights&Biases (opcional, necesita cuenta/API key)
```

## Uso

```bash
urc version
urc env launch                          # conecta con el editor de Unity abierto (pulsa Play)
urc env launch --executable build.exe --no-graphics   # conecta con un build headless

urc env create maze-v1 --build-path builds/maze.exe   # declara un entorno en urc.yaml
urc env list                                           # entornos declarados
urc env describe maze-v1                               # build_path/bridge_options/parameters/curriculum
urc env launch --env maze-v1                           # usa el build_path/bridge_options del entorno

urc config show                                          # config resuelta (defaults de la librería)
urc config show --project urc.yaml --experiment exp.yaml # + config de proyecto y experimento
urc config show --set hyperparameters.learning_rate=1e-4 # + override puntual (repetible)
urc config validate --project urc.yaml                   # valida sin imprimir
urc config diff urc.yaml otro.yaml                        # diferencias entre dos configs

urc doctor                              # Python, GPU/CUDA, qué extras opcionales hay instalados
urc init mi-proyecto                    # crea un proyecto nuevo (urc.yaml + experiments/)

urc train                                              # entrena con los defaults (mlagents + sb3-ppo)
urc train --set training.max_steps=100000              # override puntual
urc train --set algo=sb3-sac                            # SAC en vez de PPO (solo acciones continuas)
urc train --set env=maze-v1                             # usa build_path/bridge_options/curriculum de maze-v1
urc train --experiment experiments/exp1.yaml --resume runs/default/checkpoint_50000_steps.zip

urc algo list                           # nombres disponibles (built-in + plugins en ./plugins/)
urc algo info sb3-ppo                   # descripción de un algoritmo registrado

urc eval runs/default/checkpoint_50000_steps.zip --episodes 20   # reward medio, éxito, duración
urc eval runs/default/checkpoint_50000_steps.zip --success-threshold 10   # éxito = reward >= 10
urc compare runs/default/checkpoint_50000_steps.zip otro/checkpoint.zip   # compara dos evals
urc compare runs/default                # compara usando el eval más reciente de esa carpeta
urc record runs/default/checkpoint_50000_steps.zip --episodes 3   # trayectoria a .jsonl

urc visualize                           # TensorBoard sobre runs/ (o la carpeta que le pases)
urc train --set training.progress_bar=true   # barra de progreso en vivo en la terminal
urc train --set logging.backend=wandb --set logging.project=mi-proyecto   # Weights&Biases
```

¿Quieres un bridge en otro lenguaje (no Python)? [PROTOCOL.md](PROTOCOL.md) especifica el
protocolo completo (líneas JSON por stdio o socket) y [`examples/csharp_bridge/`](examples/csharp_bridge)
trae una implementación de referencia real en C#, verificada contra `urc train` de verdad:

```yaml
bridge: external
bridge_options:
  command: ["ruta/a/tu-bridge.exe"]
```

`urc eval`/`urc record` no necesitan que repitas `--project`/`--set`: `urc train` deja un
`run_info.json` junto a los checkpoints con el bridge/algoritmo usados, y lo usan automáticamente
(se puede seguir sobreescribiendo con `--set` si hace falta). `urc record` graba la trayectoria
(observación/acción/recompensa por paso) en `.jsonl`, no un vídeo de píxeles — ver ROADMAP, Fase 8,
para el porqué.

Para ver al agente entrenando en tiempo real no hace falta nada especial: conecta `urc train` al
editor de Unity (bridge `mlagents`, sin `--no-graphics`) y verás la escena normal en la ventana del
editor mientras entrena; y `urc visualize` corriendo en otra terminal actualiza los gráficos de
TensorBoard en vivo mientras el entrenamiento avanza.

Añadir un algoritmo propio no requiere tocar el código de `urc`: basta con dejar un `.py` en
`./plugins/` del proyecto que registre una clase con `@algorithms.register("mi-algo")` (ver
`urc.core.contracts.AlgorithmBackend`); `urc train --set algo=mi-algo` y `urc algo list` lo
recogen automáticamente.

Un entorno con currículo y domain randomization se declara en `urc.yaml` así:

```yaml
env: maze-v1
environments:
  maze-v1:
    build_path: builds/maze.exe
    bridge_options:
      no_graphics: true
    parameters:            # domain randomization estática, aplicada una vez al empezar
      wind: 0.2
    curriculum:             # progresión automática según la recompensa media reciente
      - parameters: { difficulty: 0.1 }
        min_reward: 0.5
        min_episodes: 10
      - parameters: { difficulty: 0.9 }
```

## Desarrollo

```bash
ruff check .        # lint
pytest               # tests
pre-commit install   # activa los hooks de pre-commit

pip install -e ".[docs]"
mkdocs serve         # sitio de documentación en local, con recarga en caliente
mkdocs build --strict   # falla si hay enlaces rotos — así se verifica antes de cada release
```

## Empaquetado

```bash
python -m pip install build twine
python -m build              # genera dist/*.whl y dist/*.tar.gz
python -m twine check dist/*  # valida la metadata para PyPI sin subir nada
```

El paquete está listo para publicarse (metadata completa, `twine check` en verde, probado
instalando el wheel en un venv limpio), pero **no se ha publicado a PyPI todavía** — es una
decisión deliberada: publicar de verdad necesita una cuenta/API key de PyPI que solo el
mantenedor debe usar (`python -m twine upload dist/*` cuando decidas hacerlo).

## Contribuir

Los issues y pull requests son bienvenidos — ver [CONTRIBUTING.md](CONTRIBUTING.md) para cómo
configurar el entorno, qué ejecutar antes de abrir un PR, y dónde encaja cada tipo de cambio
(bridge, algoritmo, ejemplo, documentación). Este proyecto sigue un
[Código de Conducta](CODE_OF_CONDUCT.md).
