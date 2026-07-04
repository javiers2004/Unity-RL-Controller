# Ejemplo: reach-target (sin Unity)

El ejemplo más sencillo para probar `urc` de extremo a extremo **sin necesitar Unity instalado**.
`env_server.py` es un entorno de juguete 1D: un agente tiene que moverse hasta un punto objetivo
(a la izquierda o a la derecha), hablando el mismo protocolo que cualquier bridge de `urc` (ver
[PROTOCOL.md](../../PROTOCOL.md)) sobre un socket TCP normal y corriente.

Con los hiperparámetros por defecto, PPO aprende la tarea de forma consistente en unos ~20.000
timesteps (unos segundos en CPU): política óptima ≈ 8 pasos por episodio, recompensa media ≈ 7.2,
100% de éxito.

## Probarlo

Terminal 1 — arranca el entorno:

```bash
cd examples/toy_reach_target
python env_server.py --port 9000
```

Terminal 2 — entrena, evalúa y compara:

```bash
cd examples/toy_reach_target
urc train                                        # usa este urc.yaml
urc eval runs/default/checkpoint_20000_steps.zip --episodes 30
urc record runs/default/checkpoint_20000_steps.zip --episodes 1
urc visualize                                    # TensorBoard sobre runs/
```

Si `urc eval` reporta una tasa de éxito cercana al 100% y ~8 pasos de media, todo el pipeline
(bridge -> entrenamiento -> checkpoint -> evaluación) está funcionando correctamente.

## Qué mirar en el código

- `env_server.py`: un servidor TCP de ~90 líneas que habla el protocolo de `urc` sin importar
  nada de `urc` — es la forma más corta de ver el protocolo en acción.
- `urc.yaml`: la config completa de este ejemplo (bridge `socket`, algoritmo `sb3-ppo` con los
  defaults).

Para probar con SAC en vez de PPO (la tarea tiene acciones continuas, así que SAC también vale):

```bash
urc train --set algo=sb3-sac --set hyperparameters.learning_starts=100
```
