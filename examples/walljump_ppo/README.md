# Ejemplo: Unity ML-Agents "WallJump" + PPO

Un agente que tiene que saltar (a veces empujando un bloque) para superar un muro y llegar a la
meta. Se eligió como segundo ejemplo con Unity real (además de
[`unity_basic_ppo/`](../unity_basic_ppo)) para tener un entorno con más movimiento a la hora de
probar el [vídeo automático del progreso de entrenamiento](../unity_basic_ppo/README.md#vídeo-automático-del-progreso-de-entrenamiento).

Acciones `MultiDiscrete` (moverse + saltar), observación con tres sensores (un vector base más dos
sensores de raycast para detectar el muro) — `MLAgentsBridge` los concatena automáticamente en un
único vector (ver ROADMAP, Fase 3).

## Requisitos y preparación (un poco más que Basic)

WallJump viene, como casi todos los ejemplos oficiales de ML-Agents, preparado para entrenar 24
copias en paralelo con `mlagents-learn` — hace falta reducirlo a una sola copia para usarlo con
`urc` (que solo soporta un agente activo a la vez). También su script cambia de comportamiento
según la dificultad del episodio, lo que hay que fijar a una sola variante. Con el proyecto oficial
de ML-Agents ya clonado (ver [`unity_basic_ppo/README.md`](../unity_basic_ppo/README.md#requisitos)):

1. Abre `Assets/ML-Agents/Examples/WallJump/Scenes/WallJump.unity`.
2. En el Hierarchy verás ~24 objetos "WallJumpArea" (numerados). **Bórralos todos menos uno**
   (desactivarlos no basta — verificado que a veces siguen contando como agentes activos igualmente,
   causa no determinada; borrarlos si funciona). Selecciona el rango con Shift+clic y pulsa Supr.
3. Edita `Assets/ML-Agents/Examples/WallJump/Scripts/WallJumpAgent.cs`: en `Initialize()` y
   `OnEpisodeBegin()`, cambia `m_Configuration = Random.Range(0, 5);` por
   `m_Configuration = Random.Range(0, 2);`. Sin esto, con el tiempo suficiente el entorno acaba
   registrando dos comportamientos distintos (`SmallWallJump` y `BigWallJump` — muro alto con
   bloque que empujar), y `MLAgentsBridge` solo soporta uno.
4. Copia [`unity/UrcVideoRecorder/UrcVideoRecorder.cs`](../../unity/UrcVideoRecorder/UrcVideoRecorder.cs)
   a `Assets/UrcRecorder/` (si no lo has hecho ya para Basic) y arrástralo sobre la cámara de la
   escena (p. ej. "PlayerCam").
5. Guarda la escena con Ctrl+S.

## Probarlo

Igual que Basic: arranca `urc train` primero (para que el puerto esté escuchando) y dale a Play en
Unity después — si lo haces al revés, la conexión falla por timeout.

```bash
cd examples/walljump_ppo
urc train                                  # usa este urc.yaml (grabación activada por defecto)
urc train --set training.max_steps=10000   # tanda corta para probar rápido
```

Verificado (2026-07-05): 10.240 pasos, duración media de episodio bajando de 89.8 a 53.9 pasos —
aprendizaje real, aunque WallJump necesita bastantes más pasos que Basic para converger del todo.
Vídeo de progreso generado en `runs/default/video/training_progress.mp4` (1.253 fotogramas / 125 s
en esa tanda de prueba).
