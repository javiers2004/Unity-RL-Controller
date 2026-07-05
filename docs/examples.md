# Ejemplos

Cuatro proyectos completos en [`examples/`](https://github.com/javiers2004/Unity-RL-Controller/tree/master/examples),
cada uno con su propio `urc.yaml` y README. Entre los cuatro cubren mapas distintos, bridges
distintos (socket, subproceso, Unity) y algoritmos distintos (PPO, SAC).

## `toy_reach_target`

Sin Unity: un entorno de juguete 1D servido por un socket TCP en ~90 líneas de Python. El punto
de partida recomendado — es lo que sigue la [guía rápida](quickstart.md). PPO, verificado
aprendiendo la tarea de verdad (100% de éxito, política óptima de 8 pasos).

[Ver README →](https://github.com/javiers2004/Unity-RL-Controller/tree/master/examples/toy_reach_target)

## `csharp_bridge`

Bridge de referencia en C# (compilado con el compilador que ya trae .NET Framework en Windows,
sin instalar el SDK de .NET), lanzado como subproceso. Demuestra el
[protocolo out-of-process](protocol.md) funcionando en un lenguaje que no es Python. Entrena con
SAC (acciones continuas).

[Ver README →](https://github.com/javiers2004/Unity-RL-Controller/tree/master/examples/csharp_bridge)

## Unity ML-Agents Basic + PPO

El ejemplo con Unity real: la escena `Basic` del proyecto oficial de ML-Agents (observación de 20
dimensiones, acción discreta de 3 valores) — la misma que se usó para verificar `MLAgentsBridge`
contra Unity de verdad. Necesita Unity Hub + el proyecto de ejemplo de ML-Agents clonado aparte
(instrucciones en el README del ejemplo).

[Ver README →](https://github.com/javiers2004/Unity-RL-Controller/tree/master/examples/unity_basic_ppo)

## Unity ML-Agents WallJump + PPO

Un segundo ejemplo con Unity real, con más movimiento: la escena oficial `WallJump` (el agente
salta para superar un muro), acciones `MultiDiscrete`, tres sensores de observación concatenados
en uno. Se usó también para verificar el
[vídeo automático del progreso de entrenamiento](https://github.com/javiers2004/Unity-RL-Controller/blob/master/examples/unity_basic_ppo/README.md#vídeo-automático-del-progreso-de-entrenamiento)
contra un entorno más exigente que Basic. Necesita algo más de preparación que Basic (instrucciones
en el README del ejemplo).

[Ver README →](https://github.com/javiers2004/Unity-RL-Controller/tree/master/examples/walljump_ppo)
