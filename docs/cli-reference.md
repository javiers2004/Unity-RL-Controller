# Referencia de comandos

Todos los comandos aceptan `--help` para ver sus opciones completas (`urc train --help`, etc.).
Los que resuelven configuración (`train`, `eval`, `record`, `config *`, `env list/describe/launch`)
aceptan `--project`/`--experiment`/`--set` para la resolución jerárquica de config (ver [Roadmap](roadmap.md), sección "Sistema de configuración").

## Proyecto

| Comando | Qué hace |
|---|---|
| `urc init <nombre>` | Crea un proyecto nuevo (`urc.yaml` + `experiments/`). |
| `urc doctor` | Python, GPU/CUDA, qué dependencias opcionales (`mlagents`, `sb3`, `wandb`) hay instaladas. |

## Configuración

| Comando | Qué hace |
|---|---|
| `urc config show` | Imprime la config final resuelta (YAML). |
| `urc config validate` | Valida sin imprimir; exit code 1 y mensaje legible si falla. |
| `urc config diff <a> <b>` | Diferencias entre dos configs ya resueltas. |

## Entornos

| Comando | Qué hace |
|---|---|
| `urc env list` | Entornos declarados en `environments:` de la config. |
| `urc env describe <nombre>` | `build_path`/`bridge_options`/`parameters`/`curriculum` de un entorno. |
| `urc env create <nombre> [--build-path RUTA]` | Añade un entorno nuevo a `urc.yaml`. |
| `urc env launch [--env NOMBRE] [--executable RUTA] [--no-graphics]` | Conecta con Unity (editor o build) y muestra los specs de obs/acción. Solo `mlagents`, no bridges arbitrarios. |

## Algoritmos

| Comando | Qué hace |
|---|---|
| `urc algo list` | Nombres disponibles (built-in + plugins en `./plugins/`). No fuerza ninguna dependencia opcional. |
| `urc algo info <nombre>` | Descripción del algoritmo (sí importa el módulo). |

## Entrenamiento y evaluación

| Comando | Qué hace |
|---|---|
| `urc train [--resume RUTA]` | Entrena. Deja un `run_info.json` junto a los checkpoints. |
| `urc eval <checkpoint> [--episodes N] [--success-threshold X]` | Reward medio, éxito, duración; guarda `eval_<checkpoint>.json`. |
| `urc compare <path...>` | Compara resultados de `eval` entre checkpoints/runs. |
| `urc record <checkpoint> [--episodes N] [--output RUTA]` | Trayectoria a `.jsonl` (no vídeo — ver [Roadmap, Fase 8](roadmap.md)). |
| `urc visualize [logdir] [--port N]` | Lanza TensorBoard sobre los logs de entrenamiento. |

`urc eval`/`urc record` no necesitan que repitas la config de `urc train`: usan el `run_info.json`
automáticamente (y lo puedes seguir sobreescribiendo con `--set`).

## Overrides de config (`--set`)

Cualquier campo de `UrcConfig` (ver [Roadmap](roadmap.md), sección "Sistema de configuración") se puede sobreescribir con
`--set clave.anidada=valor`, repetible:

```bash
urc train --set bridge=socket --set bridge_options.port=9000 --set hyperparameters.learning_rate=1e-4
```

Los valores se interpretan como YAML (`true`/`false`, números, listas `[1,2,3]`, etc.), con un
arreglo especial para que `3e-4` se lea como float (ver Fase 4 del [Roadmap](roadmap.md) para el
porqué).
