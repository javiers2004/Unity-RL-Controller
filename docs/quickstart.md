# Guía rápida

Vas a entrenar tu primer agente en unos 5 minutos, **sin necesitar Unity instalado** — usando el
ejemplo autocontenido `toy_reach_target`. La conexión con Unity real se explica al final.

## 1. Instalar

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (git-bash); .venv\Scripts\activate en cmd/PowerShell
pip install -e ".[dev,sb3]"     # sb3: entrenamiento + TensorBoard + barra de progreso
```

Comprueba que todo está en orden:

```bash
urc doctor
```

## 2. Arrancar un entorno de juguete

En una terminal:

```bash
cd examples/toy_reach_target
python env_server.py --port 9000
```

Esto deja un servidor TCP escuchando — es el "mapa" que vas a entrenar. Es un servidor normal y
corriente que habla el [protocolo de `urc`](protocol.md); no depende de `urc` para nada.

## 3. Entrenar

En otra terminal, desde la misma carpeta (`examples/toy_reach_target`):

```bash
urc train
```

Verás el bridge y el algoritmo usados, y dónde se guardan los checkpoints. Con los defaults del
ejemplo (20.000 timesteps), tarda unos segundos en CPU.

## 4. Evaluar

```bash
urc eval runs/default/checkpoint_20000_steps.zip --episodes 20
```

Deberías ver una tasa de éxito cercana al 100% — la tarea es sencilla y PPO la aprende de forma
consistente. Si quieres ver la trayectoria paso a paso:

```bash
urc record runs/default/checkpoint_20000_steps.zip --episodes 1
```

## 5. Visualizar

```bash
urc visualize
```

Levanta TensorBoard sobre `runs/`. Si lo dejas corriendo mientras entrenas otra vez en paralelo,
verás los gráficos actualizarse en vivo.

## Siguiente paso: conectar con Unity de verdad

1. Instala el extra de ML-Agents: `pip install -e ".[dev,mlagents]"` (necesita tener Unity Hub +
   un Editor instalado).
2. Clona el proyecto de ejemplo oficial de ML-Agents y ábrelo con Unity Hub:
   ```bash
   git clone --depth 1 https://github.com/Unity-Technologies/ml-agents.git
   ```
   (la carpeta a abrir en Unity Hub es `ml-agents/Project`, no `DevProject`/`PerformanceProject`).
3. Abre la escena `Assets/ML-Agents/Examples/Basic/Scenes/Basic.unity` y pulsa **Play**.
4. `urc env launch` para comprobar la conexión, o sigue directamente el ejemplo
   [`examples/unity_basic_ppo`](examples.md) para entrenar contra ella.

## Y ahora...

- ¿Quieres tu propio mapa, con currículo o domain randomization? Mira la sección de config en el
  [Roadmap](roadmap.md) (sección "Sistema de configuración") y el tutorial de
  [currículo](tutorials/curriculum.md).
- ¿Tu simulador no está en Python? Mira el tutorial de
  [escribir un bridge en otro lenguaje](tutorials/write-a-bridge.md).
- ¿Quieres tu propio algoritmo? Mira el tutorial de
  [escribir un algoritmo propio](tutorials/write-an-algorithm.md).
