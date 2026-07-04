# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/). Este proyecto
todavía no ha publicado ninguna versión estable (sigue en `0.x`): mientras tanto, cada fase del
[ROADMAP](https://github.com/javiers2004/Unity-RL-Controller/blob/master/ROADMAP.md) se documenta aquí como si fuera una entrada de versión, aunque no se haya
publicado a PyPI todavía (ver la Fase 11 sobre el estado de la publicación).

## [Unreleased]

### Fase 11 — Calidad, empaquetado y publicación
- `LICENSE` (MIT) y metadata de empaquetado completa (clasificadores, URLs del proyecto).
- Paquete verificado con `python -m build` + `twine check` + instalación en un venv limpio.
- Sitio de documentación pública (mkdocs).
- Ejemplos end-to-end: entorno de juguete autocontenido (sin Unity), bridge en C#, y Unity real.
- CI ampliado con un build headless real de Unity descargado desde un Release de GitHub.

### Fase 10 — Extensibilidad multi-lenguaje real
- `PROTOCOL.md`: especificación completa del protocolo out-of-process (bridges en cualquier
  lenguaje) — transporte, formato de líneas, métodos, tipos.
- Bridge de referencia real en C# (`examples/csharp_bridge/`), compilado con el compilador de
  .NET Framework de Windows (sin instalar el SDK), verificado contra un `ExternalProcessBridge`
  real y usado para entrenar un PPO completo.

### Fase 9 — Visualización y observabilidad
- Integración real con TensorBoard (`logging.backend: tensorboard`, el default) y con
  Weights&Biases (`logging.backend: wandb`, extra opcional `sb3`/`wandb`).
- `urc visualize`: lanza TensorBoard apuntando a los logs de un run.
- Barra de progreso en vivo en terminal (`training.progress_bar`).

### Fase 8 — Evaluación y benchmarking
- `urc eval <checkpoint>`: reward medio, tasa de éxito, duración, guardado en `eval_*.json`.
- `urc compare <path...>`: compara métricas entre checkpoints/runs.
- `urc record <checkpoint>`: graba la trayectoria de un episodio en `.jsonl`.
- `run_info.json` junto a los checkpoints, para que `eval`/`record` no obliguen a repetir la
  config usada al entrenar.

### Fase 7 — Entornos y mapas
- `EnvironmentSpec` completo: `build_path`, `bridge_options`, `parameters` (domain randomization
  estática), `curriculum` (progresión automática por episodios).
- `urc env list / describe / create`, y `urc env launch --env <nombre>`.
- `BridgeAdapter.set_parameters()`: nuevo método (no abstracto, retrocompatible) implementado de
  verdad en `MLAgentsBridge` (side channel de ML-Agents) y en los bridges JSON-RPC.

### Fase 6 — Algoritmos intercambiables
- `sb3-sac`: segundo algoritmo real (solo acciones continuas), compartiendo mecánica de
  entrenar/reanudar/checkpoint con `sb3-ppo` vía una base común.
- Los plugins de terceros (`./plugins/`, entry points) se cargan de verdad desde el CLI
  (`urc train`, `urc algo list/info`) — existían desde la Fase 2 pero nadie los invocaba.
- `urc algo list / info`.

### Fase 5 — CLI mínimo viable de entrenamiento
- `urc train`, uniendo bridge + config + algoritmo. Algoritmo por defecto: **PPO de
  Stable-Baselines3** (`sb3-ppo`) sobre un wrapper Gymnasium (`BridgeGymEnv`), no el trainer
  nativo de ML-Agents (que no compone con nuestro `BridgeAdapter`).
- Checkpointing a disco y reanudar entrenamiento (`--resume`).
- `urc doctor` (Python, GPU/CUDA, dependencias opcionales) y `urc init` (scaffolding de proyecto).
- `Registry.register_lazy`: bridges/algoritmos con dependencias opcionales pesadas se importan
  bajo demanda.

### Fase 4 — Sistema de configuración
- `UrcConfig` (pydantic v2) con resolución jerárquica: defaults de la librería → `urc.yaml` del
  proyecto → YAML de experimento → overrides `--set`.
- `urc config show / validate / diff`.

### Fase 3 — Bridge por defecto: Unity ML-Agents
- `MLAgentsBridge`, envolviendo `mlagents_envs.UnityEnvironment` — verificado contra Unity real
  (editor y modo headless).
- `SocketBridge` (TCP), además del `ExternalProcessBridge` (subproceso) de la Fase 2: dos
  transportes de ejemplo compartiendo el mismo protocolo JSON-RPC por líneas.
- `urc env launch`.

### Fase 2 — Contratos y sistema de plugins
- Contratos `BridgeAdapter`, `Policy`, `AlgorithmBackend`, `EnvironmentSpec`.
- `Registry` genérico (registro por nombre), con carga de plugins in-process (entry points,
  carpeta de plugins) y out-of-process (protocolo JSON-RPC por líneas, cualquier lenguaje).

### Fase 1 — Esqueleto del repositorio
- Estructura del proyecto, CLI mínimo (Typer), packaging editable, lint (ruff), CI básico.
