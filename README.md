# Unity-RL-Controller (`urc`)

Librería + CLI para controlar entrenamientos de Reinforcement Learning en Unity desde la terminal: bridge Unity↔código, algoritmos, mapas, hiperparámetros, evaluación y visualización, todo con comandos simples y componentes intercambiables.

El diseño completo y el plan de desarrollo por fases están en [ROADMAP.md](ROADMAP.md).

> **Estado actual**: Fases 1-4 completadas (esqueleto, contratos/plugins, bridge de ML-Agents
> verificado contra Unity real, sistema de configuración jerárquico). Siguiente: Fase 5 (CLI
> mínimo viable de entrenamiento).

## Instalación (desarrollo)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (git-bash) — usa .venv\Scripts\activate en cmd/PowerShell
pip install -e ".[dev]"
```

Para usar el bridge por defecto contra Unity (ML-Agents), instala también el extra `mlagents`
(requiere tener Unity + un proyecto con el paquete `com.unity.ml-agents` para probarlo de verdad):

```bash
pip install -e ".[dev,mlagents]"
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
```

## Desarrollo

```bash
ruff check .        # lint
pytest               # tests
pre-commit install   # activa los hooks de pre-commit
```
