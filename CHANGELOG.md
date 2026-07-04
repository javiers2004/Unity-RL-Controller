# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/). Este proyecto
todavía no ha publicado ninguna versión estable (sigue en `0.x`): mientras tanto, cada fase del
[ROADMAP](https://github.com/javiers2004/Unity-RL-Controller/blob/master/ROADMAP.md) se documenta aquí como si fuera una entrada de versión, aunque no se haya
publicado a PyPI todavía (ver la Fase 11 sobre el estado de la publicación).

## [Unreleased]

### Ampliación posterior a la Fase 8 — vídeo automático del progreso de entrenamiento
- Nuevo `unity/UrcVideoRecorder/UrcVideoRecorder.cs`: script para la escena de Unity que captura
  fotogramas (`Texture2D.ReadPixels`) y expone `Time.timeScale` a Python.
- Nuevo side channel `RecordingControlChannel` + `MLAgentsBridge.set_time_scale`/`start_recording`.
- Nuevo `RecordingCallback` (SB3): cámara rápida durante el grueso del entrenamiento, ventanas a
  velocidad normal cada N pasos, episodios finales normales, y ensamblado a `.mp4` con `imageio`.
- Nueva sección de config `recording:` (`enabled`, `fast_forward_speed`,
  `normal_speed_every_n_steps`, `final_episodes`, `fps`, `keep_frames`) y nuevo extra `video`.
- **Verificado de verdad contra Unity real** (escena Basic): 453 fotogramas / 45.3 s con contenido
  genuino, tras encontrar y arreglar cuatro bugs reales — el más interesante: la escena Basic
  resetea cada episodio recargando la escena entera, destruyendo y recreando el componente
  constantemente; arreglado con campos `static` que sobreviven a la recarga. Detalle completo en
  `ROADMAP.md`, Fase 8.
- Documentado en `examples/unity_basic_ppo/README.md`.

### Fase 12 — Pulido final y comunidad
- README: badges reales (CI, Unity integration, Docs, licencia, versión de Python) y una sección
  "Demo" con una transcripción de terminal real y verificada (`urc train`/`urc eval` contra
  `toy_reach_target`).
- `CONTRIBUTING.md`: entorno de desarrollo, checks antes de un PR, dónde encaja cada tipo de
  cambio, y cómo aportar una demo visual (grabación de pantalla, fuera del alcance de este repo).
- `CODE_OF_CONDUCT.md`: adaptación del Contributor Covenant v2.1.

### Fase 11 — Calidad, empaquetado y publicación
- `LICENSE` (MIT) y metadata de empaquetado completa (clasificadores, URLs del proyecto).
- Paquete verificado con `python -m build` + `twine check` + instalación en un venv limpio.
- Sitio de documentación pública (mkdocs).
- Ejemplos end-to-end: entorno de juguete autocontenido (sin Unity), bridge en C#, y Unity real.
- CI ampliado con un build headless real de Unity descargado desde un Release de GitHub —
  verificado en `ubuntu-latest` (`1 passed`), tras arreglar un SIGSEGV al inicializar el player
  (faltaba un servidor X; solucionado con `Xvfb`) y un `UnityPlayer.so` ausente del build subido.
- Fix: `_resolve_executable_path` en `core/rpc.py` — rutas relativas con separador fallaban al
  lanzar subprocesos en instalaciones de Python de Microsoft Store en Windows.
- Fix: `MLAgentsBridge.step()` devolvía `reward` como `np.float32` en vez de `float`, incumpliendo
  el contrato de `StepResult`.

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
