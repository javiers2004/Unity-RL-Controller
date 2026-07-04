# Currículo y domain randomization

`EnvironmentSpec` soporta dos formas de parametrizar un entorno desde config, sin tocar código
(ver Fase 7 del [Roadmap](../roadmap.md) para el diseño completo):

- **`parameters`**: valores fijos, aplicados una vez al empezar a entrenar (domain randomization
  "estática" — cada entorno declarado puede tener los suyos).
- **`curriculum`**: una progresión automática de "lessons", cada una con sus propios parámetros y
  un umbral de recompensa para pasar a la siguiente.

Ambos se aplican vía `BridgeAdapter.set_parameters(dict)` — un método del contrato que, por
defecto, no hace nada (así que si tu bridge no lo soporta, simplemente no tiene efecto, no rompe
nada). `MLAgentsBridge` lo implementa de verdad vía el `EnvironmentParametersChannel` de
ML-Agents; los bridges JSON-RPC (socket/subproceso) lo mandan como un método más del protocolo.

## Declararlo en `urc.yaml`

```yaml
env: maze-v1
environments:
  maze-v1:
    build_path: builds/maze.exe
    parameters:              # aplicado una vez, al principio
      wind: 0.2
    curriculum:               # progresión automática
      - parameters: { difficulty: 0.1 }
        min_reward: 0.5
        min_episodes: 10
      - parameters: { difficulty: 0.5 }
        min_reward: 0.8
        min_episodes: 10
      - parameters: { difficulty: 0.9 }   # última lesson: sin umbral, no se avanza más
```

Cada lesson necesita `min_episodes` episodios completos con recompensa media ≥ `min_reward` antes
de avanzar a la siguiente. La última lesson no necesita `min_reward` (no hay a dónde avanzar).

## Cómo lo consume tu entorno

Del lado del bridge, `set_parameters({"difficulty": 0.5})` te llega como una llamada más — en
ML-Agents, tu script de C# lo lee con:

```csharp
float difficulty = Academy.Instance.EnvironmentParameters.GetWithDefault("difficulty", 0.1f);
```

En un bridge JSON-RPC propio (ver el
[tutorial de bridges](write-a-bridge.md)), es el método `set_parameters` del protocolo — tu
programa decide qué hacer con esos valores (o ignorarlos, si no te interesan).

## Verificarlo

`urc train` imprime qué entorno está usando; si quieres confirmar que las lessons se están
aplicando de verdad, lo más directo es instrumentar tu propio `set_parameters` para que registre
los valores que recibe mientras entrenas, o mirar los logs de TensorBoard/wandb si tu entorno
expone la dificultad actual como una métrica.
