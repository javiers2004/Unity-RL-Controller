# Unity-RL-Controller — Roadmap y diseño

> **Qué es este documento**: la especificación completa del proyecto, dividida en fases secuenciales (pipeline). Es un documento vivo: cada fase tiene una lista de tareas con checkboxes (`- [ ]`) que se van marcando conforme se completan. Las decisiones aún no cerradas están marcadas con `[DECISIÓN PENDIENTE]`.
>
> **Última actualización**: 2026-07-04

---

## 1. Visión

Hoy, conectar Unity con código de Reinforcement Learning (crear el mapa, definir observaciones/acciones, levantar el entrenamiento, ajustar hiperparámetros, evaluar, visualizar resultados) es posible pero **cuesta mucho integrar** cada pieza por separado.

El objetivo de este proyecto es construir una **librería + CLI** que convierta todo ese proceso en comandos simples de terminal, sin sacrificar la posibilidad de personalizar cualquier pieza del sistema. Idea central en una frase:

> **Todo tiene un valor por defecto que funciona sin configurar nada, y todo se puede reemplazar sin tocar el core.**

Esto aplica a los tres ejes que ya se han decidido:

| Eje | Por defecto (fácil) | Personalizable |
|---|---|---|
| Conexión Unity ↔ código (**Bridge**) | Unity ML-Agents Toolkit (`mlagents_envs`, gRPC ya resuelto) | Cualquier bridge propio (sockets, gRPC custom, shared memory...) que cumpla el contrato |
| Algoritmo de entrenamiento (**Backend**) | Trainer nativo de ML-Agents (PPO, ya incluido, cero dependencias extra) | Stable-Baselines3, RLlib, o implementación propia |
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
bridge: mlagents          # o "custom:mi_bridge.py"
algo: mlagents-ppo        # o "sb3-ppo", "custom:mi_algo.py"
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
│   │   ├── mlagents_bridge.py (default)
│   │   └── external_bridge.py (protocolo out-of-process)
│   ├── algorithms/
│   │   ├── mlagents_ppo.py    (default)
│   │   └── sb3_adapter.py
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

### Fase 3 — Bridge por defecto: Unity ML-Agents ✅ (código) / ⏳ (verificación con Unity real)
- [x] Wrapper de `mlagents_envs.UnityEnvironment` implementando `BridgeAdapter`
      (`src/urc/bridges/mlagents_bridge.py`, registrado como `"mlagents"`)
- [x] Soporte modo headless (`no_graphics=True`) + múltiples instancias en paralelo
      (`worker_id`/`base_port`) — expuesto tal cual de `UnityEnvironment`
- [x] Modo "editor" para debug interactivo: `file_name=None` conecta contra el editor
      abierto en vez de lanzar un build
- [x] `urc env launch` (`--executable`, `--no-graphics`, `--worker-id`, `--seed`, `--timeout`)
      para lanzar/verificar conexión — **pendiente de probar contra Unity real**, ver nota abajo
- [x] Segundo bridge mínimo de ejemplo: `SocketBridge` (TCP, `src/urc/bridges/socket_bridge.py`),
      además del `ExternalProcessBridge` de la Fase 2 — ambos comparten protocolo
      (`JsonLineRpcClient`/`JsonLineBridge` en `core/rpc.py`), demostrando que el contrato es
      intercambiable con distintos transportes (stdio, socket, y el gRPC interno de ML-Agents)

**Limitación conocida y deliberada de `MLAgentsBridge`**: solo soporta un behavior con un único
agente activo y un único sensor de observación (el caso de los entornos de ejemplo simples de
ML-Agents, como Basic o GridWorld). Si el entorno tiene más de un agente/behavior/sensor activo,
lanza `NotImplementedError` con un mensaje explícito en vez de comportarse de forma incorrecta en
silencio. Multi-agente queda para cuando el contrato lo necesite explícitamente.

**Gotcha real encontrado**: `mlagents-envs` 0.28.0 (la última en PyPI) trae bindings de protobuf
generados con una versión antigua; con `protobuf>=3.21` falla al importar. Se fija
`protobuf<3.21` en el extra `mlagents` de `pyproject.toml`.

**Pendiente de verificación real** (requiere Unity instalado, ver nota más abajo): todo lo de
esta fase está probado con un `UnityEnvironment` falso en los tests (`test_mlagents_bridge.py`),
pero `urc env launch` todavía no se ha ejecutado contra un Unity de verdad. Es el primer paso a
hacer en cuanto el usuario tenga Unity + un entorno de ejemplo de ML-Agents disponibles.

### Fase 4 — Sistema de configuración
- [ ] Loader de YAML jerárquico (defaults → proyecto → experimento → CLI)
- [ ] Validación con schema (pydantic) y errores legibles
- [ ] `urc config show / validate / diff`

### Fase 5 — CLI mínimo viable de entrenamiento
- [ ] `urc train` uniendo bridge + config + algoritmo por defecto (PPO nativo de ML-Agents)
- [ ] Checkpointing a disco + reanudar entrenamiento (`--resume`)
- [ ] `urc doctor` (diagnóstico de instalación: Unity, Python, GPU, dependencias)
- [ ] `urc init <nombre>` (scaffolding de proyecto nuevo)

### Fase 6 — Algoritmos intercambiables
- [ ] Adapter para Stable-Baselines3 como segundo backend
- [ ] Mecanismo para que el usuario registre su propio algoritmo (plugin in-process u out-of-process)
- [ ] `urc algo list / info`
- [ ] Gestión de hiperparámetros vía `--set clave=valor` y archivos de hiperparámetros reutilizables

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
