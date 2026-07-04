# Contribuir a Unity-RL-Controller

Gracias por el interés. Este proyecto nació como herramienta personal pero aspira a ser útil como
librería open-source — las contribuciones (código, documentación, ejemplos, informes de bugs) son
bienvenidas.

## Antes de programar

- Para cambios pequeños (typos, docs, bugs claros y acotados): abre un PR directamente.
- Para cambios grandes (nuevo bridge/algoritmo *built-in*, cambios de arquitectura): abre antes un
  issue para discutir el enfoque — evita trabajo desperdiciado si el diseño no encaja con
  [ROADMAP.md](ROADMAP.md), que documenta las decisiones de diseño ya tomadas y el porqué.

## Configurar el entorno

```bash
git clone https://github.com/javiers2004/Unity-RL-Controller.git
cd Unity-RL-Controller
python -m venv .venv
source .venv/Scripts/activate   # Windows (git-bash) — usa .venv\Scripts\activate en cmd/PowerShell
pip install -e ".[dev,all]"
pre-commit install
```

## Antes de abrir el PR

```bash
ruff check .
pytest
mkdocs build --strict   # solo si tocas docs/, ROADMAP.md, PROTOCOL.md o CHANGELOG.md
```

CI (`.github/workflows/ci.yml`) ejecuta lint + tests en cada push/PR. `unity-integration.yml` solo
se dispara si el cambio toca el bridge de ML-Agents o su test de integración, y necesita un build
real de Unity (se descarga de un GitHub Release — ver [ROADMAP.md](ROADMAP.md), Fase 11, para el
porqué de ese diseño).

## Dónde encaja tu cambio

- **Nuevo bridge** (otro motor de simulación, otro lenguaje): implementa
  `urc.core.contracts.BridgeAdapter` — ver
  [docs/tutorials/write-a-bridge.md](docs/tutorials/write-a-bridge.md). Si es para un lenguaje
  distinto de Python, sigue además [PROTOCOL.md](PROTOCOL.md) (protocolo out-of-process).
- **Nuevo algoritmo**: implementa `AlgorithmBackend` y regístralo con
  `@algorithms.register(...)` — ver
  [docs/tutorials/write-an-algorithm.md](docs/tutorials/write-an-algorithm.md). Si no depende de
  una librería ya soportada (Stable-Baselines3), probablemente encaja mejor como plugin en
  `./plugins/` de un proyecto que como algoritmo *built-in* de `urc`.
- **Documentación**: vive en `docs/*.md` + `mkdocs.yml`. `ROADMAP.md`/`PROTOCOL.md`/`CHANGELOG.md`
  viven en la raíz del repo y se transcluyen en el sitio (`pymdownx.snippets`) — no los dupliques
  dentro de `docs/`.
- **Ejemplos**: en `examples/`. Cada uno debe ser autocontenido (su propio `urc.yaml` +
  `README.md`) y, si es razonable, verificado de verdad — entrenar y comprobar que aprende la
  tarea, no solo que el pipeline no lanza excepciones.

## Estilo de commits

Mensajes cortos en modo imperativo, explicando el *porqué* cuando no sea obvio. No hace falta
seguir Conventional Commits ni ninguna convención estricta.

## Grabar una demo visual

El [README](README.md) trae una transcripción de terminal real como demo, pero no un GIF/vídeo del
agente entrenando dentro del editor de Unity — grabar pantalla no es algo que se pueda automatizar
desde aquí. Si quieres aportar uno:

1. Usa [`examples/unity_basic_ppo/`](examples/unity_basic_ppo) (o cualquier otro ejemplo) con el
   bridge `mlagents` sin `--no-graphics`, para ver la ventana del editor mientras entrena.
2. Grábalo con cualquier herramienta (ScreenToGif, OBS, `peek`...), unos segundos bastan.
3. Guarda el archivo en `docs/assets/` y enlázalo desde `README.md`/`docs/index.md`.
4. Mantén el tamaño del archivo razonable (unos pocos MB) — son GIFs de demo, no grabaciones completas.

## Código de conducta

Este proyecto sigue el [Código de Conducta](CODE_OF_CONDUCT.md). Al participar, se espera que lo
respetes.
