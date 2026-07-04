# Unity-RL-Controller (`urc`)

Librería + CLI para controlar entrenamientos de Reinforcement Learning en Unity desde la terminal: bridge Unity↔código, algoritmos, mapas, hiperparámetros, evaluación y visualización, todo con comandos simples y componentes intercambiables.

El diseño completo y el plan de desarrollo por fases están en [ROADMAP.md](ROADMAP.md).

> **Estado actual**: Fases 1-6 completadas: esqueleto, contratos/plugins, bridge de ML-Agents
> verificado contra Unity real, configuración jerárquica, `urc train` de extremo a extremo (PPO o
> SAC de Stable-Baselines3 sobre cualquier bridge, checkpointing y `--resume`), y algoritmos de
> terceros vía `./plugins/` conectados de verdad al CLI. Siguiente: Fase 7 (entornos y mapas).

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
pip install -e ".[dev,sb3]"       # solo el algoritmo de entrenamiento
```

## Uso

```bash
urc version
urc env launch                          # conecta con el editor de Unity abierto (pulsa Play)
urc env launch --executable build.exe --no-graphics   # conecta con un build headless

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
urc train --experiment experiments/exp1.yaml --resume runs/default/checkpoint_50000_steps.zip

urc algo list                           # nombres disponibles (built-in + plugins en ./plugins/)
urc algo info sb3-ppo                   # descripción de un algoritmo registrado
```

Añadir un algoritmo propio no requiere tocar el código de `urc`: basta con dejar un `.py` en
`./plugins/` del proyecto que registre una clase con `@algorithms.register("mi-algo")` (ver
`urc.core.contracts.AlgorithmBackend`); `urc train --set algo=mi-algo` y `urc algo list` lo
recogen automáticamente.

## Desarrollo

```bash
ruff check .        # lint
pytest               # tests
pre-commit install   # activa los hooks de pre-commit
```
