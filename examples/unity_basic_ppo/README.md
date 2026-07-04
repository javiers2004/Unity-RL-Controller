# Ejemplo: Unity ML-Agents "Basic" + PPO

El ejemplo con Unity de verdad — usa la escena `Basic` del proyecto oficial de ML-Agents (la
misma que se usó para verificar `MLAgentsBridge` contra Unity real en la Fase 3 del
[ROADMAP](../../ROADMAP.md)): observación de 20 dimensiones, 1 acción discreta de 3 valores
(izquierda / quieto / derecha).

## Requisitos

- Unity Hub + Unity 2022.3 LTS o 6000.x, con el proyecto de ejemplo de ML-Agents clonado:
  ```bash
  git clone --depth 1 https://github.com/Unity-Technologies/ml-agents.git
  ```
  Ábrelo con Unity Hub apuntando a la carpeta `Project/` de ese clon (no `DevProject/` ni
  `PerformanceProject/`, que están casi vacíos).
- `pip install -e ".[dev,mlagents,sb3]"` (o `.[dev,all]`).

## Probarlo

1. En Unity, abre `Assets/ML-Agents/Examples/Basic/Scenes/Basic.unity`.
2. Pulsa **Play**.
3. En la terminal, desde esta carpeta:
   ```bash
   urc train
   ```
   `urc` se conecta al editor (`bridge_options: {}` = sin build, como `urc env launch` sin
   `--executable`) y entrena PPO contra la escena en marcha. Verás la ventana del editor
   funcionando en tiempo real — es la forma más directa de "live view" (ver Fase 9 del ROADMAP).
4. Cuando termine (o lo pares con Ctrl+C, dejando algún checkpoint guardado):
   ```bash
   urc eval runs/default/checkpoint_10000_steps.zip --episodes 10
   ```
   Para evaluar necesitas la escena en Play otra vez (`urc eval` también se conecta al editor).

## Usar un build headless en vez del editor

Si exportas un build headless (`File > Build Settings`, plataforma actual, sin `-batchmode` hace
falta especificarlo aparte — `urc` ya pasa `--batchmode`/`-nographics` internamente vía
`no_graphics`), añade la ruta en `bridge_options`:

```yaml
bridge_options:
  file_name: "ruta/al/build/Basic.exe"
  no_graphics: true
```

Así no hace falta tener el editor abierto ni pulsar Play — útil para dejar entrenamientos largos
corriendo sin la ventana del editor delante.
