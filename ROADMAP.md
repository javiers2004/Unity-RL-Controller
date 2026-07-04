# Unity-RL-Controller — Roadmap y diseño

> **Qué es este documento**: la especificación completa del proyecto, dividida en fases secuenciales (pipeline). Es un documento vivo: cada fase tiene una lista de tareas con checkboxes (`- [ ]`) que se van marcando conforme se completan. Las decisiones aún no cerradas están marcadas con `[DECISIÓN PENDIENTE]`.
>
> **Última actualización**: 2026-07-04 (Fase 12 completada — todas las fases planificadas hechas)

---

## 1. Visión

Hoy, conectar Unity con código de Reinforcement Learning (crear el mapa, definir observaciones/acciones, levantar el entrenamiento, ajustar hiperparámetros, evaluar, visualizar resultados) es posible pero **cuesta mucho integrar** cada pieza por separado.

El objetivo de este proyecto es construir una **librería + CLI** que convierta todo ese proceso en comandos simples de terminal, sin sacrificar la posibilidad de personalizar cualquier pieza del sistema. Idea central en una frase:

> **Todo tiene un valor por defecto que funciona sin configurar nada, y todo se puede reemplazar sin tocar el core.**

Esto aplica a los tres ejes que ya se han decidido:

| Eje | Por defecto (fácil) | Personalizable |
|---|---|---|
| Conexión Unity ↔ código (**Bridge**) | Unity ML-Agents Toolkit (`mlagents_envs`, gRPC ya resuelto) | Cualquier bridge propio (sockets, gRPC custom, shared memory...) que cumpla el contrato |
| Algoritmo de entrenamiento (**Backend**) | PPO de Stable-Baselines3 (`sb3-ppo`), vía un wrapper Gymnasium sobre el bridge — ver nota de la Fase 5 sobre por qué no es el trainer nativo de ML-Agents | RLlib, implementación propia, o (más adelante) invocar `mlagents-learn` como backend "nativo" |
| Lenguaje de la CLI/librería | Python | Cualquier lenguaje, vía plugins "out-of-process" con protocolo definido |

Y el propósito del proyecto es **doble**: debe ser cómodo de usar día a día como herramienta de investigación personal, y a la vez estar construido con el rigor (tests, packaging, docs, versionado) de una librería open-source publicable.

---

## 2. Principios de diseño

1. **Contrato antes que implementación.** Cada pieza reemplazable (bridge, algoritmo, entorno) se define primero como una interfaz mínima. La implementación por defecto es solo "una implementación más" de esa interfaz.
2. **Todo es plugin.** Bridges, algoritmos, entornos/mapas y hasta partes del propio CLI se registran en un *registry* central bajo un nombre, y se seleccionan por config o por flag de CLI (`--bridge`, `--algo`, `--env`).
3. **Config por encima de código.** El comportamiento de un experimento (qué bridge, qué algoritmo, qué hiperparámetros, qué mapa) se debe poder describir entero en un archivo de config versionable, no en código Python disperso.
4. **CLI como superficie única de control.** Si algo se puede hacer con la librería, se debe poder hacer con un comando. La API de Python es la base; el CLI es una capa fina encima.
5. **Cero fricción para empezar, sin techo para crecer.** `urc init` + `urc train` deben funcionar en minutos con los defaults. Pero un usuario avanzado debe poder sustituir cualquier pieza sin forkear el proyecto.
6. **Multi-lenguaje real, no solo de palabra.** La forma de lograr "personalizable a cualquier lenguaje" es definir un protocolo neutral (stdin/stdout + JSON-RPC, o gRPC) para plugins "out-of-process", además del camino rápido in-process en Python.

---

## 3. Glosario rápido

- **Bridge**: componente que habla con Unity (lanza el build, intercambia observaciones/acciones/recompensas). Hoy en día el `UnityEnvironment` de ML-Agents es el bridge de referencia.
- **Backend / Algoritmo**: implementación del algoritmo de RL (PPO, SAC, etc.) que consume lo que da el Bridge y produce una política entrenada.
- **EnvironmentSpec**: metadatos de un mapa/escena de Unity: espacio de observación, espacio de acción, parámetros configurables del mapa, curriculum, etc.
- **Registry**: tabla interna que mapea un nombre (`"mlagents"`, `"sb3-ppo"`, `"maze-v1"`) a una clase/plugin concreto.
- **Plugin in-process**: código Python que implementa directamente la interfaz (rápido, mismo proceso).
- **Plugin out-of-process**: proceso externo (en cualquier lenguaje) que habla el protocolo definido por el contrato vía stdio/gRPC (para el caso "quiero mi propio lenguaje").
- **Run / Experimento**: una ejecución concreta de entrenamiento, con su config, checkpoints, logs y métricas asociadas.

---

## 4. Arquitectura de alto nivel

```
                          ┌───────────────────────────┐
                          │            CLI            │  urc train / eval / visualize ...
                          └─────────────┬─────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │      Config resuelta       │  defaults + proyecto + experimento + overrides CLI
                          └─────────────┬─────────────┘
                                        │
        ┌───────────────────┬──────────┼──────────┬───────────────────┐
        ▼                   ▼          ▼          ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────────┐
│ Bridge Registry│   │ Algo Registry │   │ Env Registry  │   │ Logging/Viz Registry│
│  (mlagents,    │   │ (mlagents-ppo,│   │ (mapas Unity, │   │ (tensorboard, wandb,│
│   custom...)   │   │  sb3, custom) │   │  procedurales)│   │  dashboard propio)  │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘   └──────────┬─────────┘
        │                   │                   │                      │
        └─────────┬─────────┴─────────┬─────────┘                      │
                   ▼                   ▼                                │
           ┌───────────────────────────────────┐                       │
           │           Training Loop            │───────────────────────┘
           │  (orquesta bridge + algo + env)    │
           └─────────────────┬───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Runs / Checkpoints  │
                  └─────────────────────┘
```

Cada caja con "Registry" es un punto de extensión: acepta implementaciones in-process (Python) u out-of-process (cualquier lenguaje) siempre que cumplan el contrato correspondiente.

### 4.1 Contratos mínimos ✅ (implementados en Fase 2, `src/urc/core/contracts.py`)

- `BridgeAdapter`: `reset()`, `step(action)`, `observation_spec()`, `action_spec()`, `close()`.
- `AlgorithmBackend`: `train(bridge, env_spec, config) -> Policy`, `load(checkpoint) -> Policy`.
- `Policy`: `predict(observation) -> action`. *(Refinado respecto al borrador inicial: `predict`
  vive en `Policy`, no en `AlgorithmBackend`, para poder tener varias políticas/checkpoints
  cargados a la vez — necesario para `urc compare` en la Fase 8 — sin que compartan estado.)*
- `EnvironmentSpec`: metadatos declarativos (no lógica) — nombre, ruta al build, specs de obs/acción, parámetros configurables, curriculum opcional.

El registro de componentes (`src/urc/core/registry.py`) expone `bridges`, `algorithms` y
`environments`, cada uno resoluble por nombre string (`registry.get("nombre")` /
`registry.create("nombre", ...)`). Los plugins in-process se cargan vía entry points
(`urc.bridges`, `urc.algorithms`, `urc.environments`) o desde una carpeta local
(`urc.core.plugins.load_plugins_from_dir`). Los plugins out-of-process (para escribirlos en
cualquier lenguaje) hablan un protocolo JSON-RPC por stdio (`urc.core.rpc.StdioRpcClient`),
implementado de referencia en `ExternalProcessBridge` (`src/urc/bridges/external_bridge.py`,
registrado como `"external"`) y validado con un subproceso real en los tests.

---

## 5. Sistema de configuración

Jerarquía de resolución (cada nivel sobreescribe al anterior):

1. **Defaults de la librería** (empaquetados con `urc`, ej. `ppo` con hiperparámetros razonables).
2. **Config del proyecto** (`urc.yaml` en la raíz del repo del usuario).
3. **Config del experimento** (`experiments/mi-experimento.yaml`).
4. **Overrides de línea de comandos** (`--set lr=3e-4`, `--bridge custom-socket`).

Formato: YAML, validado con schema (pydantic o similar) para dar errores claros en vez de tracebacks. Ejemplo ilustrativo:

```yaml
bridge: mlagents          # o "socket", "external", "custom:mi_bridge.py"
algo: sb3-ppo             # o "custom:mi_algo.py"
env: maze-v1
hyperparameters:
  learning_rate: 3.0e-4
  gamma: 0.99
  batch_size: 1024
training:
  max_steps: 2_000_000
  checkpoint_every: 50_000
logging:
  backend: tensorboard     # o "wandb", "none"
```

Comandos asociados: `urc config show`, `urc config validate`, `urc config diff <a> <b>`.

---

## 6. Diseño de la CLI

Nombre de comando propuesto: `urc` (**U**nity **R**L **C**ontroller) — `[DECISIÓN PENDIENTE]` confirmar o cambiar.

| Comando | Descripción |
|---|---|
| `urc init <nombre>` | Crea un proyecto nuevo con estructura y config por defecto |
| `urc doctor` | Diagnostica el entorno: Unity instalado, versión de Python, GPU, dependencias, builds disponibles |
| `urc env list / describe / launch` | Lista, inspecciona o lanza un mapa/entorno registrado |
| `urc bridge list / set / info` | Gestiona qué bridge está activo |
| `urc algo list / info` | Lista algoritmos registrados y sus hiperparámetros disponibles |
| `urc train [--bridge] [--algo] [--env] [--set k=v]` | Lanza un entrenamiento |
| `urc eval <run>` | Evalúa un checkpoint N episodios y reporta métricas |
| `urc compare <run1> <run2> ...` | Compara métricas entre runs/experimentos |
| `urc record <run>` | Graba vídeo/replay de un episodio con la política entrenada |
| `urc visualize [<run>]` | Levanta TensorBoard u otro dashboard configurado |
| `urc plugin list / add / remove` | Gestiona plugins de terceros (bridges, algoritmos, entornos) |
| `urc config show / validate / diff` | Gestiona la configuración jerárquica |

---

## 7. Estructura de repositorio propuesta

```
unity-rl-controller/
├── ROADMAP.md                 (este documento)
├── PROTOCOL.md                 (especificación del protocolo out-of-process, Fase 10)
├── pyproject.toml
├── src/urc/
│   ├── cli/                   comandos (uno por subcomando) + _shared.py (opciones/carga común)
│   ├── core/
│   │   ├── contracts.py       BridgeAdapter, AlgorithmBackend, EnvironmentSpec
│   │   ├── registry.py        sistema de registro de plugins (register/register_lazy/set)
│   │   ├── environments.py    EnvironmentSpec desde config -> registry (ver Fase 7)
│   │   ├── evaluation.py      EpisodeResult/EvalResult + run_episodes() (Fase 8)
│   │   ├── runs.py            run_info.json de un run entrenado (Fase 8)
│   │   └── jsonutil.py        json_safe(): numpy -> tipos nativos, sin depender de numpy
│   ├── bridges/
│   │   ├── mlagents_bridge.py (default, carga perezosa)
│   │   ├── external_bridge.py (protocolo out-of-process, subproceso)
│   │   └── socket_bridge.py   (protocolo out-of-process, TCP)
│   ├── algorithms/
│   │   ├── sb3_base.py        (SB3Backend + _build_logging: tensorboard/wandb, ver Fase 9)
│   │   ├── sb3_ppo.py         (default: PPO, admite acciones discretas y continuas)
│   │   ├── sb3_sac.py         (alternativa: SAC, solo acciones continuas)
│   │   ├── gym_bridge.py      (BridgeAdapter -> entorno Gymnasium)
│   │   └── curriculum.py      (CurriculumCallback para SB3)
│   └── config/                loader + schema + resolución jerárquica
├── unity/                     proyecto(s) de Unity con las escenas/mapas
├── examples/
│   └── csharp_bridge/         bridge de referencia en C# (Fase 10, ver PROTOCOL.md)
├── docs/                      documentación pública (mkdocs)
└── tests/
```

---

## 8. Pipeline de desarrollo por fases

> Cada fase produce algo funcional y verificable antes de pasar a la siguiente. No hace falta terminar el 100% de una fase para asomarse a la siguiente, pero sí tener el "core" de cada fase cerrado.

### Fase 0 — Fundamentos y decisiones base
- [ ] Confirmar nombre del proyecto y del comando CLI (`urc` u otro)
- [ ] Elegir licencia (MIT/Apache-2.0, dado el objetivo open-source)
- [ ] Prototipo mínimo: verificar que ML-Agents responde bien como bridge por defecto (smoke test manual)
- [ ] Redactar el contrato (interfaz) exacto de `BridgeAdapter`, `AlgorithmBackend`, `EnvironmentSpec`
- [ ] Elegir gestor de entorno/paquetes Python (`uv` recomendado, o `poetry`/`venv+pip`)

### Fase 1 — Esqueleto del repositorio ✅
- [x] `git init` + primer commit
- [x] Estructura de carpetas (`src/` layout) según sección 7
- [x] `pyproject.toml` con metadata, dependencias base, entry point del CLI
- [x] Instalación editable (`pip install -e .`) funcionando
- [x] Linter/formatter (`ruff`) + pre-commit hooks
- [x] CI básico (lint + tests en cada push)
- [x] README corto que enlaza a este ROADMAP

### Fase 2 — Contratos y sistema de plugins ✅
- [x] Clases base abstractas: `BridgeAdapter`, `Policy`, `AlgorithmBackend`, `EnvironmentSpec`
- [x] `Registry` genérico (registrar por nombre string, resolver desde config)
- [x] Soporte de plugins in-process (Python, vía `entry_points` o carpeta de plugins)
- [x] Soporte de plugins out-of-process: protocolo JSON-RPC sobre stdio (decisión tomada, ver
      sección 10) + `ExternalProcessBridge` como implementación de referencia
- [x] Implementación *fake/no-op* de cada contrato + tests unitarios sobre los contratos
      (15 tests, incluido un round-trip real contra un subproceso)

### Fase 3 — Bridge por defecto: Unity ML-Agents ✅
- [x] Wrapper de `mlagents_envs.UnityEnvironment` implementando `BridgeAdapter`
      (`src/urc/bridges/mlagents_bridge.py`, registrado como `"mlagents"`)
- [x] Soporte modo headless (`no_graphics=True`) + múltiples instancias en paralelo
      (`worker_id`/`base_port`) — expuesto tal cual de `UnityEnvironment`
- [x] Modo "editor" para debug interactivo: `file_name=None` conecta contra el editor
      abierto en vez de lanzar un build
- [x] `urc env launch` (`--executable`, `--no-graphics`, `--worker-id`, `--seed`, `--timeout`)
      para lanzar/verificar conexión — **verificado contra Unity real** (ver nota abajo)
- [x] Segundo bridge mínimo de ejemplo: `SocketBridge` (TCP, `src/urc/bridges/socket_bridge.py`),
      además del `ExternalProcessBridge` de la Fase 2 — ambos comparten protocolo
      (`JsonLineRpcClient`/`JsonLineBridge` en `core/rpc.py`), demostrando que el contrato es
      intercambiable con distintos transportes (stdio, socket, y el gRPC interno de ML-Agents)

**Limitación conocida y deliberada de `MLAgentsBridge`**: solo soporta un behavior con un único
agente activo y un único sensor de observación (el caso de los entornos de ejemplo simples de
ML-Agents, como Basic o GridWorld). Si el entorno tiene más de un agente/behavior/sensor activo,
lanza `NotImplementedError` con un mensaje explícito en vez de comportarse de forma incorrecta en
silencio. Multi-agente queda para cuando el contrato lo necesite explícitamente.

**Gotcha real encontrado #1**: `mlagents-envs` 0.28.0 (la última en PyPI) trae bindings de
protobuf generados con una versión antigua; con `protobuf>=3.21` falla al importar. Se fija
`protobuf<3.21` en el extra `mlagents` de `pyproject.toml`.

**Gotcha real encontrado #2**: `action_spec()` reportaba `dtype="float32"` para acciones
discretas (heredado del default del dataclass), cuando internamente ya se construían como
`int32` para la llamada a `ActionTuple`. Solo se detectó al probar contra Unity real, no lo
cubría el `UnityEnvironment` falso de los tests — corregido y añadido a la aserción del test.

**Verificado contra Unity real** (2026-07-04): `urc env launch` conectado al editor con la escena
`Basic.unity` de ML-Agents (Unity 6000.0.77f1) — reportó `Observaciones: shape=(20,)
dtype=float32` y `Acciones: shape=(1,) dtype=int32 discreto=True`, coincidiendo exactamente con
la especificación conocida del entorno Basic (observación one-hot de 20 posiciones, 1 acción
discreta). Instalación usada: Unity Hub 3.19.3 + Unity 6000.0.77f1, proyecto de ejemplo oficial
clonado de `Unity-Technologies/ml-agents` (carpeta `Project/`, no `DevProject/` ni
`PerformanceProject/`, que están casi vacíos).

### Fase 4 — Sistema de configuración ✅
- [x] Loader de YAML jerárquico (defaults → proyecto → experimento → CLI),
      `urc/config/loader.py::resolve_config`, con deep-merge (los dicts anidados se
      fusionan, no se sustituyen wholesale)
- [x] Validación con schema (pydantic v2, `urc/config/schema.py::UrcConfig`) y errores
      legibles (`ConfigError`, lista "campo: motivo" en vez de traceback de pydantic)
- [x] `urc config show / validate / diff`

**Defaults empaquetados vs defaults del schema**: `UrcConfig()` "a pelo" (sin pasar por
`resolve_config`) tiene `hyperparameters={}` — los valores reales (`learning_rate`, `gamma`,
`batch_size`) viven en `src/urc/config/defaults.yaml`, la capa 1 de la jerarquía. Son
intencionalmente distintos: el schema define la *forma*, `defaults.yaml` define los *valores*
por defecto reales, siguiendo el principio de "config por encima de código" (sección 2).

**Gotcha real encontrado**: `yaml.safe_load("3e-4")` devuelve el *string* `"3e-4"`, no el float
`0.0003` — la spec YAML 1.1 exige un punto decimal en la mantisa para reconocer notación
científica. Como es la forma más natural de escribir un learning rate en un `--set`, se sustituyó
por un parser de escalares propio (`int` → `float` → `bool`/`null` → YAML como último recurso)
que sí lo entiende. Sin este fix, `--set hyperparameters.learning_rate=3e-4` habría fallado en
silencio (guardaría un string donde se espera un número).

**Nota sobre `algo: mlagents-ppo`**: es un nombre provisional en `defaults.yaml`; todavía no
existe ningún backend de algoritmo registrado (eso es la Fase 6). El sistema de config no valida
que `bridge`/`algo`/`env` existan de verdad en sus registries — esa comprobación ocurre al
resolver/instanciar desde config (Phase 5+), reutilizando el `KeyError` con opciones disponibles
que ya da `Registry.get()` (Fase 2).

### Fase 5 — CLI mínimo viable de entrenamiento ✅
- [x] `urc train` uniendo bridge + config + algoritmo por defecto
- [x] Checkpointing a disco + reanudar entrenamiento (`--resume`)
- [x] `urc doctor` (Python, GPU/CUDA, dependencias opcionales instaladas o no)
- [x] `urc init <nombre>` (scaffolding de proyecto nuevo)

**Pivote importante respecto al borrador inicial**: el algoritmo por defecto ya NO es "el trainer
nativo de ML-Agents" (`mlagents-learn`), sino **PPO de Stable-Baselines3** (`sb3-ppo`), entrenado a
través de un wrapper Gymnasium (`BridgeGymEnv`) sobre *cualquier* `BridgeAdapter`. Motivo: el
trainer nativo de ML-Agents no compone con nuestro contrato — gestiona su propia conexión a Unity
internamente y no acepta un `BridgeAdapter` externo, así que usarlo como "el default" habría
significado que la primera vez que se entrena algo, ni siquiera se ejercita la arquitectura de
bridges que se ha construido en las Fases 2-3. SB3 sobre `BridgeGymEnv` sí funciona con
*cualquier* bridge (mlagents, socket, subproceso) sin acoplarse a Unity en absoluto — es una
validación real de que el contrato funciona, no solo un wrapper cosmético. Invocar
`mlagents-learn` como backend alternativo (para quien quiera self-play/currículo nativos de
ML-Agents) queda como posibilidad futura, no descartada, solo pospuesta.

**Nuevas piezas de arquitectura que introdujo esta fase** (no estaban en el diseño original):
- `Registry.register_lazy(nombre, módulo, install_hint=...)`: permite que `bridges.get("mlagents")`
  o `algorithms.get("sb3-ppo")` importen su módulo bajo demanda la primera vez que se piden de
  verdad, sin forzar sus dependencias pesadas (grpc/protobuf, torch/gymnasium) a quien no las usa.
  Si la importación falla, el error incluye el `pip install "urc[...]"` correcto.
- `BridgeGymEnv` (`src/urc/algorithms/gym_bridge.py`): adapta cualquier `BridgeAdapter` a la
  interfaz estándar de Gymnasium, para que cualquier librería de ese ecosistema (SB3, RLlib...)
  pueda entrenar contra él sin saber nada de Unity.
- `ActionSpec.discrete_branches`: campo nuevo (opcional) en el contrato — la Fase 3 solo
  reportaba `shape`/`discrete` para mostrarlos en `urc env launch`, pero construir un espacio Gym
  discreto de verdad (`Discrete(n)` / `MultiDiscrete([...])`) necesita la cardinalidad de cada
  rama, no solo cuántas ramas hay. Extensión retrocompatible (default `None`).
- `bridge_options` / `output_dir` en `UrcConfig`: `bridge_options` son los kwargs que se pasan
  literalmente al constructor del bridge elegido (p. ej. `bridge_options.file_name` para
  `mlagents`, `bridge_options.host`/`.port` para `socket`) — evita acoplar `urc train` a los
  parámetros concretos de cada bridge antes de que exista `EnvironmentSpec` completo (Fase 7).

**Tres bugs reales encontrados al probar `urc train` de verdad** (ninguno lo detectaban los tests
con dobles, porque nunca se había ejercitado el camino completo bridge→Gym→SB3):
1. `_action_space` construía un `Box(-inf, +inf)` para acciones continuas — SB3 exige límites
   finitos en el espacio de *acciones* (no en el de observaciones). Arreglado con `[-1, 1]`,
   la convención habitual de ML-Agents para acciones continuas.
2. SB3 pasa las acciones como `numpy.ndarray`, que `json.dumps` no sabe serializar — rompía
   `SocketBridge`/`ExternalProcessBridge` (que hablan JSON por líneas). Arreglado con `_json_safe`
   en `core/rpc.py`, que usa duck-typing (`hasattr(valor, "tolist")`) para no tener que añadir
   numpy como dependencia dura de esos dos bridges (su gracia es depender de cero extras).
3. `Registry.create(name, **kwargs)` (Fase 2) chocaba si el propio plugin tenía un parámetro
   llamado igual que uno interno — ya arreglado entonces, mencionado aquí porque es la misma
   categoría de bug (solo se ve entrenando de verdad, no con mocks).

**Verificado manualmente de extremo a extremo** (2026-07-04): `urc train` contra un servidor TCP
de juguete (bridge `socket`), con hiperparámetros minúsculos para que corra en segundos —
checkpoints guardados correctamente en disco, y `--resume` continuando el contador de timesteps
sin reiniciarlo. Automatizado como test en `tests/test_cli_train.py`.

### Fase 6 — Algoritmos intercambiables ✅
- [x] Segundo backend real y distinto: **SAC de Stable-Baselines3** (`sb3-sac`). Comparte mecánica
      de entrenar/reanudar/checkpoint con `sb3-ppo` vía una base común (`algorithms/sb3_base.py`,
      `SB3Backend`) — solo cambia la clase de algoritmo de SB3. SAC no admite acciones discretas
      (limitación real del algoritmo): se comprueba explícitamente y se rechaza con un mensaje
      claro (`_check_action_space`) en vez de dejar que falle dentro de SB3. El candidato
      "invocar `mlagents-learn` como proceso externo" sigue disponible para más adelante si hace
      falta self-play/currículo nativos de ML-Agents — no se ha hecho ahora por complejidad y
      riesgo de dependencias frente al beneficio (ver Fase 5 sobre el pivote a SB3).
- [x] Mecanismo para que el usuario registre su propio algoritmo: **ya existía desde la Fase 2**
      (`load_plugins_from_dir`/`load_entry_point_plugins`) pero **nunca se llamaba desde el
      CLI** — hueco real cerrado en esta fase con `load_all_plugins()`, invocado al principio de
      `urc train` y `urc algo list/info`. Antes de esta fase, un plugin de terceros no tenía
      ninguna forma de llegar a registrarse en una sesión real de `urc`.
- [x] `urc algo list / info` (`src/urc/cli/algo.py`) — `list` no importa nada (no fuerza
      dependencias opcionales solo por listar nombres), `info` sí, al pedir la descripción real.
- [x] ~~Gestión de hiperparámetros vía `--set clave=valor`~~ ya cubierto desde la Fase 4/5.

**Bug real encontrado escribiendo el test del plugin custom**: `load_plugins_from_dir` re-ejecutaba
el archivo entero cada vez que se llamaba, sin comprobar si ya estaba cargado. Como `urc train` y
`urc algo list` llaman ambos a `load_all_plugins()`, invocarlos dos veces en el mismo proceso (tal
cual pasa en los tests, y pasaría en cualquier sesión que combine varios comandos) intentaba
re-registrar el mismo nombre dos veces → `ValueError`. Arreglado haciéndolo idempotente por ruta
absoluta (`_plugin_module_name`, hash de la ruta resuelta): si el archivo ya se cargó en este
proceso, no se vuelve a importar. De paso corrige otro problema latente: dos proyectos con un
`plugins/mi_algo.py` cada uno habrían colisionado en el mismo nombre de módulo sintético.

### Fase 7 — Entornos y mapas ✅
- [x] `EnvironmentSpec` completo: `build_path`, `bridge_options`, `parameters` (domain
      randomization estática), `curriculum` (lista de lessons). `observation_spec`/`action_spec`
      se dejan sin poblar a propósito: son responsabilidad dinámica del bridge (`bridge.reset()`
      ya las expone), no algo que declarar a mano en YAML.
- [x] `urc env list / describe / create` (`src/urc/cli/env.py`), y `urc env launch --env <nombre>`
      además (no estaba en el checklist original, pero es la extensión natural de la Fase 3 ahora
      que existen entornos registrados: usa el `build_path`/`bridge_options` del entorno, con los
      flags explícitos de siempre tomando precedencia si se pasan).
- [x] Soporte real (no un stub) de curriculum learning y domain randomization vía config:
  - `BridgeAdapter.set_parameters(dict)` — nuevo método **no abstracto** con no-op por defecto
    (extensión retrocompatible del contrato, igual que `discrete_branches` en la Fase 5): los
    bridges existentes no necesitan implementarlo para seguir siendo válidos.
  - `MLAgentsBridge.set_parameters`: lo envía a Unity de verdad vía
    `EnvironmentParametersChannel` de ML-Agents (side channel). La escena solo lo usa si su propio
    código C# lo lee explícitamente — el envío en sí funciona igual lo lea o no.
  - `JsonLineBridge.set_parameters` (socket/subproceso): un método RPC más (`"set_parameters"`).
  - `CurriculumCallback` (`src/urc/algorithms/curriculum.py`): callback real de SB3 que aplica la
    primera lesson al empezar a entrenar y avanza a la siguiente cuando la recompensa media de
    los últimos N episodios supera el umbral de la lesson. Acoplado a SB3 a propósito (es nuestro
    único bucle de entrenamiento hoy); otro backend necesitaría su propia integración.
  - `parameters` estáticos de un entorno se aplican una vez al empezar a entrenar
    (`SB3Backend.train`), antes de que el currículo (si lo hay) aplique la lesson 0.
- [ ] Generación procedural de mapas parametrizada — **no se ha hecho**: es explícitamente
      opcional en el checklist original y requiere scripting de Unity Editor (C#), fuera del
      alcance de una librería del lado Python. Queda descartada salvo que surja una necesidad
      concreta más adelante.

**Verificado manualmente de extremo a extremo** (2026-07-04): `urc env create maze-v1
--build-path builds/maze.exe`, `urc env list`, `urc env describe maze-v1` contra un `urc.yaml`
real; y `urc train` con un entorno `socket` de juguete configurado con dos *lessons* de currículo
— el servidor de juguete confirmó recibir `{"difficulty": 0.1}` al empezar y `{"difficulty": 0.9}`
tras el avance, exactamente como se diseñó. Automatizado después en
`tests/test_cli_train_curriculum.py`.

**Decisión de diseño**: `Registry` gana un método `set()` (upsert, no falla si ya existe) además
de `register()` (falla si ya existe). Un `EnvironmentSpec` se reconstruye cada vez que se resuelve
la config — no es un registro de "una vez" como un plugin de código — así que necesita semántica
de upsert, no de registro único.

**Ajuste a la estructura de repositorio propuesta (sección 7)**: la carpeta `envs/` que se había
planeado ahí no ha hecho falta. Los entornos son datos declarados en YAML (`environments:` en
`urc.yaml`), no clases Python como los bridges/algoritmos, así que no hay "implementaciones" que
guardar en una carpeta de plugins — toda la lógica cabe en `core/environments.py`. Se ha eliminado
la carpeta `envs/` vacía que quedó de la Fase 1.

### Fase 8 — Evaluación y benchmarking ✅
- [x] `urc eval <checkpoint>`: corre N episodios con la política cargada (reward medio ± std,
      duración media en pasos, tasa de éxito) y guarda el resultado en `eval_<checkpoint>.json`
      junto al checkpoint, para que `urc compare` lo pueda leer después.
- [x] `urc compare <path...>`: acepta un resultado de eval directamente, un checkpoint (busca su
      `eval_<nombre>.json` hermano) o una carpeta de run (usa el más reciente), y compara
      reward/éxito/duración en una tabla.
- [x] `urc record <checkpoint>`: graba la trayectoria (observación/acción/recompensa por paso) de
      N episodios en un `.jsonl` — ver nota abajo sobre por qué esto y no vídeo de píxeles.

**`run_info.json`, pieza nueva no prevista en el diseño original**: `urc train` ahora deja un
`run_info.json` junto a los checkpoints (bridge, `bridge_options`, algoritmo, entorno). Sin esto,
`urc eval`/`urc record` habrían obligado al usuario a repetir `--project`/`--set` con exactamente
la misma config que usó para entrenar, solo para saber qué bridge/algoritmo reconstruir — un caso
claro de fricción evitable. `resolve_config` gana un parámetro `extra_defaults` (capa de prioridad
justo por encima de los defaults de la librería) para que el `run_info.json` se pueda sobreescribir
con un `--set`/`urc.yaml` explícito si hace falta, en vez de ser inamovible.

**Qué mide "éxito"**: el contrato `BridgeAdapter`/`StepResult` no tiene un campo "éxito" — solo
`reward`/`done`/`info`. `urc eval` usa, en este orden: (1) `info["success"]` del último paso si el
entorno lo reporta, (2) si no, `--success-threshold X` (recompensa total del episodio >= X), (3) si
no hay ninguna de las dos, reporta la tasa de éxito como "N/A" en vez de inventar un criterio.

**`urc record` graba trayectorias, no vídeo de píxeles**: "vídeo/replay" en el checklist original
sugería captura de vídeo. No se ha hecho así a propósito: nuestro contrato `BridgeAdapter` no
expone el renderizado de Unity (ni `MLAgentsBridge` ni los demás bridges tienen forma de capturar
píxeles), así que producir un vídeo real exigiría scripting adicional en el lado de Unity (cámara
-> archivo) y una dependencia nueva para codificar vídeo (`opencv-python`/`imageio`), ninguna de
las cuales está en el plan actual. En su lugar, `urc record` guarda un replay estructurado
(observación/acción/recompensa por paso en `.jsonl`) — genuinamente útil para depurar o analizar
una política sin tener Unity abierto, aunque no sea "ver" el episodio. Grabar vídeo de verdad queda
anotado como posible ampliación futura si hace falta, no descartado.

**Verificado manualmente de extremo a extremo** (2026-07-04): entrené un checkpoint diminuto contra
un servidor `socket` de juguete, corrí `urc eval` sobre él sin pasarle ninguna config (usó
`run_info.json` solo), comprobé el JSON de resultado guardado, corrí `urc compare` con la ruta del
checkpoint y con la carpeta del run (ambas formas resuelven al mismo resultado), y `urc record`
generó un `.jsonl` con una línea por paso. Automatizado después en `tests/test_cli_eval.py`,
`tests/test_cli_compare.py` y `tests/test_cli_record.py`.

**Ajuste a un fixture de test compartido**: `toy_env_server` (usado desde la Fase 5) solo aceptaba
una conexión; evaluar/grabar después de entrenar contra el mismo host/puerto necesita una segunda
conexión al mismo servidor. Se cambió a aceptar conexiones en bucle — más fiel además a cómo se
comportaría un entorno real, que sigue disponible entre comandos.

### Fase 9 — Visualización y observabilidad ✅
- [x] Integración TensorBoard (default): `SB3Backend.train` pasa `tensorboard_log` a SB3 cuando
      `logging.backend == "tensorboard"` (el default) — SB3 ya sabe escribir ahí solo, no hubo que
      construir nada de logging desde cero. `logging.backend == "none"` lo desactiva.
- [x] Weights&Biases (opcional, vía extra `wandb`): `logging.backend == "wandb"` inicializa un run
      de wandb con `sync_tensorboard=True` (por eso también activa `tensorboard_log`, es como está
      pensada la propia integración oficial de wandb con SB3) y añade su `WandbCallback`. Probado
      de verdad con `WANDB_MODE=disabled` (sin red ni credenciales) — cablea correctamente, aunque
      el dashboard real de wandb.ai no se puede verificar en este entorno.
- [x] `urc visualize [logdir]`: lanza TensorBoard (vía su API embebible, `tensorboard.program`) y
      devuelve la URL, igual que el comando `tensorboard --logdir ...` de siempre.
- [x] "Live view" — implementado con tres piezas reales, no una sola "gran" funcionalidad nueva:
  1. **Barra de progreso en terminal**: `training.progress_bar: true` (nuevo campo, default
     `false` para no ensuciar salida no interactiva/CI) activa `progress_bar=True` de SB3
     (tqdm/rich), ya verificado con un entrenamiento real.
  2. **Dashboard en vivo**: `urc visualize` mientras `urc train` corre en otra terminal ya
     actualiza los gráficos en tiempo real — es como funciona TensorBoard normalmente, no hace
     falta nada especial de nuestra parte.
  3. **Ver al agente de verdad**: con el bridge `mlagents` conectado al editor de Unity (sin
     `--no-graphics`), el usuario ya ve al agente entrenando en la ventana del editor mientras
     `urc train` corre — otra vez, gratis por cómo funciona ML-Agents, no requiere código nuevo.

**Por qué no una sola función "live view"**: streaming de vídeo genérico desde cualquier bridge
habría sido mucho esfuerzo (captura de píxeles, codificación, transporte) para duplicar algo que
ML-Agents y TensorBoard ya resuelven mejor cada uno en su terreno. Documentar cómo combinarlos era
más valioso que construir una alternativa propia peor.

**Verificado manualmente de extremo a extremo** (2026-07-04): entrené con `logging.backend:
tensorboard` (default) y `training.progress_bar: true` contra un servidor `socket` de juguete —
la barra de progreso se vio en la terminal, y aparecieron archivos `events.out.tfevents.*` reales
bajo `runs/default/tensorboard/`. Lancé `_launch_tensorboard` contra esos logs reales y comprobé
con una petición HTTP real que el dashboard respondía (200 OK). Probé también el wiring de wandb
con `WANDB_MODE=disabled`. Automatizado después en `tests/test_sb3_logging.py` y
`tests/test_cli_visualize.py`.

**Ajuste a la estructura de repositorio propuesta (sección 7)**: igual que `envs/` en la Fase 7, la
carpeta `logging/` planeada ahí tampoco ha hecho falta. El logging de TensorBoard/wandb se apoya en
mecanismos ya integrados en el propio bucle de entrenamiento de SB3 (`tensorboard_log`,
`WandbCallback`), así que vive junto a `SB3Backend` en `algorithms/sb3_base.py` — es lógica
específica de SB3, no algo agnóstico de algoritmo que mereciera su propio paquete. Se ha eliminado
la carpeta `logging/` vacía que quedó de la Fase 1.

### Fase 10 — Extensibilidad multi-lenguaje real ✅
- [x] Especificación formal del protocolo out-of-process:
      [`PROTOCOL.md`](https://github.com/javiers2004/Unity-RL-Controller/blob/master/PROTOCOL.md). No se hizo
      en protobuf/JSON schema formal como decía el borrador original — es Markdown con tablas y
      ejemplos. Motivo: el protocolo (líneas JSON, `method`/`params`/`result`/`error`) es lo
      bastante simple como para que un JSON Schema formal añadiera ceremonia sin aportar claridad
      extra frente a una tabla + ejemplos, que además sirve directamente como guía de uso (ver
      siguiente punto) en el mismo documento.
- [x] Implementación de referencia en otro lenguaje: **C# real, no un stub**
      ([`examples/csharp_bridge/Program.cs`](https://github.com/javiers2004/Unity-RL-Controller/blob/master/examples/csharp_bridge/Program.cs)) — compilado con
      `csc.exe` (el compilador que ya trae .NET Framework en Windows, sin instalar el SDK de
      .NET) y verificado de extremo a extremo contra un `ExternalProcessBridge` real, incluyendo
      **entrenar un PPO completo usando el ejecutable de C# como entorno** (checkpoints y logs de
      TensorBoard reales generados a partir de ahí). Alcance: solo bridges — los algoritmos
      siguen siendo plugins de Python (aclarado explícitamente en `PROTOCOL.md`, para que no se
      lea como "puedes escribir tu algoritmo de entrenamiento en C#", que no es cierto).
- [x] Documentación "cómo escribir tu propio plugin": integrada en `PROTOCOL.md` (sección 6,
      checklist) en vez de en un documento aparte — spec y guía de uso son la misma audiencia
      leyendo en el mismo momento, separarlas solo habría fragmentado la información.

**Test automatizado real, no solo un ejemplo documentado**: `tests/test_csharp_reference_bridge.py`
compila `Program.cs` con `csc.exe` y lo conecta con un `ExternalProcessBridge` de verdad en cada
ejecución — no es un mock de "cómo se comportaría C#", es C# de verdad compilado y ejecutado. Se
salta limpiamente (`pytest.mark.skipif`) si no hay `csc.exe` disponible, p. ej. en el runner Linux
de CI, igual que ya hacíamos con `mlagents_envs`/`stable_baselines3` cuando no están instalados.

**Bug de documentación evitado**: al escribir el checklist de `PROTOCOL.md` estuve a punto de decir
que un bridge propio funciona con `urc env launch` "igual que con ML-Agents" — falso: `urc env
launch` está *hardcodeado* a `MLAgentsBridge`, no resuelve el bridge desde el registry como sí
hacen `urc train`/`urc eval`/`urc record`. Corregido antes de dejarlo escrito; es una limitación
real que queda anotada, no algo que arreglar en esta fase (`env launch` nació en la Fase 3 pensado
solo para el smoke test contra Unity, no como comando genérico).

### Fase 11 — Calidad, empaquetado y publicación ✅
- [x] Cobertura de tests (unitarios + integración con builds headless en CI) — el workflow
      `unity-integration.yml` descarga un build headless real (Linux, escena Basic de ML-Agents)
      de un GitHub Release y ejecuta `tests/test_unity_headless_integration.py` contra él de
      verdad. **`1 passed`** en CI — ver nota abajo para el proceso de depuración real que hizo
      falta para llegar ahí.
- [x] Sitio de documentación pública (mkdocs, tema Material): `mkdocs.yml` +
      `docs/{index,quickstart,cli-reference,examples}.md` + 3 tutoriales
      (`docs/tutorials/{write-a-bridge,write-an-algorithm,curriculum}.md`). `ROADMAP.md`/
      `PROTOCOL.md`/`CHANGELOG.md` se transcluyen tal cual (`pymdownx.snippets`) en vez de
      duplicarse — una sola fuente de verdad. Verificado con `mkdocs build --strict` sin avisos.
      Workflow `.github/workflows/docs.yml` listo para publicar a GitHub Pages en cada push a
      `docs/`/`mkdocs.yml`; falta que el usuario active Pages en la configuración del repo (GitHub
      Settings → Pages → Source: rama `gh-pages`) — no lo puedo hacer yo.
- [x] Versionado semántico + `CHANGELOG.md` (formato Keep a Changelog, una sección por fase).
      Empaquetado verificado de verdad: `LICENSE` (MIT), metadata completa de `pyproject.toml`
      (clasificadores, URLs), `python -m build` + `twine check` en verde, e instalación del wheel
      resultante en un venv limpio con `urc version`/`urc config show` funcionando. **No publicado
      a PyPI** — decisión explícita del usuario: dejar el paquete listo, publicar el "de verdad"
      es un paso suyo (necesita su propia cuenta/API key de PyPI).
- [x] 3 ejemplos end-to-end en `examples/`, cubriendo bridges/algoritmos/mapas distintos:
  - `toy_reach_target/`: entorno de juguete autocontenido (socket TCP, sin Unity), PPO —
    **verificado aprendiendo la tarea de verdad** (100% de éxito, 8 pasos por episodio, el óptimo).
  - `csharp_bridge/`: el bridge de referencia en C# de la Fase 10, ahora con `urc.yaml` propio,
    entrenando con **SAC** (acciones continuas) — verificado end-to-end.
  - `unity_basic_ppo/`: Unity ML-Agents real (escena Basic), PPO (acciones discretas) — preparado
    con `urc.yaml`/README, pendiente de que el usuario lo confirme manualmente cuando pueda (no
    hay Unity en esta sesión de trabajo).

**Bug real encontrado montando el ejemplo de C#**: `subprocess.Popen` con una ruta relativa que
contiene separadores de carpeta fallaba con `FileNotFoundError` en esta instalación de Python de
Microsoft Store, aunque el archivo existiera de verdad (`CreateProcess` no la resuelve contra el
directorio de trabajo igual que las APIs de archivo normales de Python). Arreglado de raíz en
`urc.core.rpc._resolve_executable_path`: convierte a absoluta cualquier ruta con separador antes
de lanzar el subproceso, dejando los nombres sueltos (`"python"`, `"node"`) intactos para que se
resuelvan por `PATH` como siempre. Sin este fix, `bridge_options.command` con rutas relativas
habría fallado de forma intermitente según la instalación de Python de cada usuario.

**Sobre "CI con builds headless de Unity"**: se investigó el enfoque "montarlo dentro de GitHub
Actions" (game-ci) y se descartó — el proyecto de ejemplo de ML-Agents referencia
`com.unity.ml-agents` con una ruta relativa (`file:../../com.unity.ml-agents`), así que haría falta
vendorizar buena parte de ese monorepo en el nuestro, más una licencia de Unity como secreto de
GitHub, más riesgo de que la imagen Docker de la versión exacta de Unity no esté disponible. Se
optó en su lugar por: el usuario exporta un build headless de Linux una vez desde su propio Editor
(Unity soporta cross-compilar a Linux sin salir de Windows, solo añadiendo el módulo "Linux Build
Support" en Unity Hub), lo sube como asset de un GitHub Release (`unity-basic-linux-v1`), y el CI
simplemente lo descarga y ejecuta un smoke test real contra él. Sin licencias de Unity en GitHub
Actions, sin vendorizar ml-agents. **Completado y verificado: `1 passed` en `ubuntu-latest`.**

Tres problemas reales aparecieron montando esto, todos arreglados de raíz (no con parches):
1. **Workflow que fallaba en silencio**: si el `.x86_64` no se encontraba dentro del zip
   descomprimido, el test se saltaba (`skipped`) en vez de fallar, y el job seguía saliendo en
   verde — una señal de CI falsa y peligrosa. Arreglado buscando el ejecutable recursivamente
   (el zip metía el build dentro de una carpeta contenedora) y con `exit 1` explícito si no
   aparece ninguno.
2. **`UnityPlayer.so` que faltaba en el build subido**: el ejecutable arrancaba y moría con
   "return code 127" / `error while loading shared libraries: UnityPlayer.so: cannot open shared
   object file`. Unity coloca ese `.so` junto al `.x86_64` en un build Linux, aparte de la carpeta
   `_Data/`; se quedó fuera del zip subido al Release la primera vez. Arreglado re-empaquetando el
   build con `UnityPlayer.so` incluido.
3. **SIGSEGV al hacer `reset()`**: con el `.so` ya presente, el motor inicializaba del todo
   (versión, físicas, `NullGfxDevice`, comunicador de ML-Agents registrado) pero el proceso moría
   con signal 11 justo después, durante el primer intercambio del comunicador. Causa: en
   `ubuntu-latest` no hay ningún servidor X, y el player de Unity en Linux necesita poder conectar
   a uno (aunque sea virtual) para terminar de inicializarse sin crashear, incluso en modo
   `-nographics`/`-batchmode`. Arreglado añadiendo un paso que instala `xvfb`, levanta un display
   virtual (`Xvfb :99 -screen 0 1024x768x24 &`) y exporta `DISPLAY=:99` antes de lanzar el test.

Un cuarto problema, ya no de infraestructura sino de contrato: una vez el test llegaba a `step()`,
fallaba con `assert False` porque `StepResult.reward` está declarado como `float` pero
`MLAgentsBridge.step()` pasaba el `np.float32` de `mlagents_envs` sin convertir — no se notaba con
los bridges de prueba (mocks con floats nativos) ni con `toy_reach_target`/`csharp_bridge` (no usan
`MLAgentsBridge`), solo saltó al ejecutar contra Unity real por primera vez. Arreglado casteando a
`float(...)` en `mlagents_bridge.py`.

Con esto verificado un par de veces seguidas, `unity-integration.yml` pasó de `workflow_dispatch`
únicamente a disparo automático en `push`/`pull_request` (restringido por `paths:` a los archivos
que de verdad afectan al bridge de ML-Agents, para no descargar el build de ~28 MB en cada push).

### Fase 12 — Pulido final y comunidad ✅
- [x] README con demo y badges: badges reales de CI/Unity integration/Docs (generados por los
      workflows existentes, no estáticos), licencia y versión de Python. Sección "Demo" con una
      transcripción de terminal real (`urc train` + `urc eval` contra `toy_reach_target`,
      re-ejecutada y verificada, no inventada). **No incluye un GIF visual del agente entrenando
      en Unity** — eso es una grabación de pantalla, no algo generable desde aquí; queda
      documentado como aportación abierta en `CONTRIBUTING.md`.
- [x] `CONTRIBUTING.md`: cómo configurar el entorno, qué ejecutar antes de un PR, dónde encaja cada
      tipo de cambio (bridge/algoritmo/ejemplo/documentación), estilo de commits, y cómo aportar
      una demo visual. `CODE_OF_CONDUCT.md`: adaptación traducida del Contributor Covenant v2.1,
      con reporte vía Security Advisory privado del repo (se evita publicar un email personal).
- [~] Recoger feedback de uso real y priorizar iteración siguiente — **no es una tarea que se
      pueda completar de una vez**, es un proceso continuo que empieza cuando haya usuarios reales
      (p. ej. tras publicar a PyPI). El mecanismo ya está listo: GitHub Issues + `CONTRIBUTING.md`
      explicando cómo reportar y dónde encaja cada propuesta; queda como trabajo permanente, no
      como checkbox cerrable.

---

## 9. Consideraciones transversales

- **Testing**: separar tests unitarios (rápidos, sin Unity) de tests de integración (requieren un build headless de Unity; correrlos en CI aparte o marcarlos como manuales/locales).
- **Rendimiento**: soportar entornos vectorizados/paralelos desde el diseño del `BridgeAdapter`, no como parche posterior.
- **Seguridad**: cargar plugins de terceros implica ejecución de código arbitrario — documentar el riesgo y, si se publica como librería pública, considerar algún tipo de sandboxing o al menos advertencias claras.
- **Compatibilidad hacia atrás**: una vez publicada la v1 del contrato de plugins, cambiarlo rompe a terceros — versionar el contrato explícitamente (`contract_version`).
- **Multiplataforma**: decidir pronto si los builds headless de Unity se soportan en Windows/Linux/ambos, ya que afecta al diseño de CI.

---

## 10. Decisiones pendientes

- `[DECISIÓN PENDIENTE]` Nombre final del proyecto/comando CLI.
- `[DECISIÓN PENDIENTE]` Licencia open-source.
- ~~`[DECISIÓN PENDIENTE]` Protocolo exacto para plugins out-of-process~~ → **Decidido en Fase 2**: JSON-RPC sobre stdio (líneas de JSON por stdin/stdout). Motivo: cualquier lenguaje sabe leer/escribir JSON por stdio sin dependencias extra, frente a gRPC que exige toolchain de protobuf en cada lenguaje. Sigue siendo "solo una implementación más" del contrato `BridgeAdapter`/`AlgorithmBackend` — si en el futuro hace falta más rendimiento, se puede añadir un adapter gRPC alternativo sin tocar el core. La especificación formal del protocolo (JSON schema) queda para la Fase 10.
- `[DECISIÓN PENDIENTE]` Soporte multiplataforma desde el día 1 o empezar solo Windows (entorno actual del usuario) y generalizar después.

---

## 11. Próximos pasos inmediatos

1. Cerrar las decisiones pendientes de la Fase 0 (aunque sea con un valor "provisional" para no bloquear).
2. Ejecutar la Fase 1 (esqueleto del repositorio) — es la base para todo lo demás.
3. Volver a este documento después de cada fase para marcar checkboxes y anotar aprendizajes o cambios de rumbo.
