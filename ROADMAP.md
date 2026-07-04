# Unity-RL-Controller — Roadmap y diseño

> **Qué es este documento**: la especificación completa del proyecto, dividida en fases secuenciales (pipeline). Es un documento vivo: cada fase tiene una lista de tareas con checkboxes (`- [ ]`) que se van marcando conforme se completan. Las decisiones aún no cerradas están marcadas con `[DECISIÓN PENDIENTE]`.
>
> **Última actualización**: 2026-07-04 (Fase 6)

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
├── pyproject.toml
├── src/urc/
│   ├── cli/                   comandos (uno por subcomando)
│   ├── core/
│   │   ├── contracts.py       BridgeAdapter, AlgorithmBackend, EnvironmentSpec
│   │   └── registry.py        sistema de registro de plugins
│   ├── bridges/
│   │   ├── mlagents_bridge.py (default, carga perezosa)
│   │   ├── external_bridge.py (protocolo out-of-process, subproceso)
│   │   └── socket_bridge.py   (protocolo out-of-process, TCP)
│   ├── algorithms/
│   │   ├── sb3_base.py        (SB3Backend: mecánica común a los backends de SB3)
│   │   ├── sb3_ppo.py         (default: PPO, admite acciones discretas y continuas)
│   │   ├── sb3_sac.py         (alternativa: SAC, solo acciones continuas)
│   │   └── gym_bridge.py      (BridgeAdapter -> entorno Gymnasium)
│   ├── envs/                  EnvironmentSpecs registrados + builds de Unity
│   ├── config/                loader + schema + resolución jerárquica
│   └── logging/               integraciones tensorboard/wandb/dashboard propio
├── unity/                     proyecto(s) de Unity con las escenas/mapas
├── examples/                  proyectos de ejemplo end-to-end
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

### Fase 7 — Entornos y mapas
- [ ] `EnvironmentSpec` completo: obs/acciones, parámetros del mapa, curriculum
- [ ] `urc env list / describe / create`
- [ ] Soporte de curriculum learning y domain randomization vía config
- [ ] (Opcional) generación procedural de mapas parametrizada

### Fase 8 — Evaluación y benchmarking
- [ ] `urc eval` (N episodios, métricas: reward medio, tasa de éxito, duración)
- [ ] `urc compare` entre runs/checkpoints
- [ ] `urc record` (vídeo/replay de episodios)

### Fase 9 — Visualización y observabilidad
- [ ] Integración TensorBoard (default) + Weights&Biases (opcional, configurable)
- [ ] `urc visualize` (levanta el dashboard configurado)
- [ ] Modo "live view" para observar al agente entrenando en tiempo real

### Fase 10 — Extensibilidad multi-lenguaje real
- [ ] Especificación formal (protobuf/JSON schema) del protocolo out-of-process
- [ ] Implementación de referencia de un plugin en otro lenguaje (ej. stub en C#) para validar el contrato cross-language
- [ ] Documentación "cómo escribir tu propio plugin en el lenguaje que quieras"

### Fase 11 — Calidad, empaquetado y publicación
- [ ] Cobertura de tests (unitarios + integración con builds headless en CI)
- [ ] Sitio de documentación pública (mkdocs) con quickstart y tutoriales
- [ ] Publicación en PyPI, versionado semántico, `CHANGELOG.md`
- [ ] 2-3 ejemplos end-to-end (mapas distintos, algoritmos distintos)

### Fase 12 — Pulido final y comunidad
- [ ] README con demos/GIFs y badges
- [ ] `CONTRIBUTING.md` + código de conducta
- [ ] Recoger feedback de uso real y priorizar iteración siguiente

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
