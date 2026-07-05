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

## Vídeo automático del progreso de entrenamiento

`urc` puede generar un `.mp4` con el progreso del entrenamiento: la mayor parte a cámara rápida,
con ventanas a velocidad normal cada N pasos para apreciar la mejora, terminando con varios
episodios normales del agente ya entrenado. Requiere dos cosas que no vienen activadas por
defecto:

1. **Instalar el extra `video`**: `pip install -e ".[dev,mlagents,sb3,video]"` (o `.[dev,all]`,
   que ya lo incluye). Trae `imageio` + un binario de `ffmpeg` portable, sin instalar nada aparte.
2. **Copiar el script C# a tu proyecto de Unity**: copia
   [`unity/UrcVideoRecorder/UrcVideoRecorder.cs`](../../unity/UrcVideoRecorder/UrcVideoRecorder.cs)
   a `Assets/UrcRecorder/UrcVideoRecorder.cs` dentro de tu clon de `ml-agents`, y arrástralo como
   componente sobre cualquier GameObject de la escena `Basic.unity` (p. ej. la Main Camera). Esto
   es un paso manual en el Editor — no hay forma de automatizarlo desde fuera de Unity.

Con eso hecho:

```bash
urc train --set recording.enabled=true --set recording.fast_forward_speed=50
```

Al terminar, el vídeo queda en `runs/default/video/training_progress.mp4`. Opciones disponibles
(`urc config show --set recording.<opción>=...`): `fast_forward_speed` (por defecto 20),
`normal_speed_every_n_steps` (1000), `final_episodes` (4), `final_time_scale` (0.25 — cámara lenta
de verdad para los episodios finales, no solo velocidad normal: como esos episodios duran poco
tiempo real, a `timeScale=1` apenas da tiempo a capturar un par de fotogramas del acercamiento y se
ve como un teletransporte en vez de un movimiento; más lento = más fotogramas del mismo
recorrido), `fps` (10 — debe coincidir con `CaptureIntervalSeconds` en `UrcVideoRecorder.cs`, ver
comentario ahí si cambias uno de los dos) y `keep_frames` (false — si es `true`, conserva también
los PNG sueltos en `video_frames/`, útil solo para depurar).

Solo funciona con el bridge `mlagents`: `BridgeAdapter` no expone el renderizado de Unity a
propósito (ver ROADMAP, Fase 8), así que capturar píxeles solo es posible desde dentro de la
propia escena. Con cualquier otro bridge, `recording.enabled=true` simplemente avisa y no hace
nada — el entrenamiento sigue igual, sin vídeo.

**Verificado de verdad**: una tanda de 5.000 pasos + 3 episodios finales produjo un vídeo de 453
fotogramas (45.3 s) con contenido genuino. Si tu escena resetea episodios recargando la escena
entera (como hace `Basic` — ver `BasicController.ResetAgent()`), `UrcVideoRecorder.cs` ya lo tiene
en cuenta (usa campos `static` que sobreviven a la recarga); si escribes tu propio script de
grabación para otra escena, ten en cuenta esa misma trampa — ver ROADMAP, Fase 8, para el porqué.
