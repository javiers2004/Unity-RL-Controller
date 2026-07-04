# Escribir un bridge en otro lenguaje

Si tu simulador no está en Python (un juego en C++, un robot con firmware en C, un entorno en
Node.js...), puedes conectarlo a `urc` sin escribir ni una línea de Python de más — solo tu
programa hablando un protocolo de líneas JSON por `stdin`/`stdout` o por un socket TCP.

La especificación completa está en **[PROTOCOL.md](../protocol.md)**; esta página es la versión
corta con un ejemplo real.

## La idea en 3 líneas

1. Tu programa lee una línea JSON `{"method": "...", "params": {...}}`.
2. Hace lo que le pidan (`reset`, `step`, `observation_spec`, `action_spec`).
3. Escribe una línea de vuelta: `{"result": ...}` o `{"error": "..."}`.

## Ejemplo real: C#

[`examples/csharp_bridge/Program.cs`](https://github.com/javiers2004/Unity-RL-Controller/blob/master/examples/csharp_bridge/Program.cs)
es un bridge completo, sin depender de `urc` ni de Unity — solo `JavaScriptSerializer` (incluido
en .NET Framework, sin NuGet). El núcleo es un `switch` sobre `method`:

```csharp
switch (method)
{
    case "reset":
        // ...devuelve la observación inicial
    case "step":
        // ...aplica la acción, devuelve observación/reward/done
    case "observation_spec":
    case "action_spec":
        // ...describen la forma de tus datos, una vez
    case "set_parameters":
        // opcional: domain randomization / currículo
}
```

Compílalo y pruébalo tú mismo — instrucciones en el
[README del ejemplo](https://github.com/javiers2004/Unity-RL-Controller/tree/master/examples/csharp_bridge).

## Conectarlo a `urc`

Como subproceso (`urc` lo lanza él solo):

```yaml
bridge: external
bridge_options:
  command: ["ruta/a/tu-programa"]
```

O como servidor TCP (tú lo arrancas aparte, `urc` se conecta):

```yaml
bridge: socket
bridge_options:
  host: 127.0.0.1
  port: 9000
```

## Antes de usarlo con `urc train`

Pruébalo directamente contra `ExternalProcessBridge`/`SocketBridge` primero:

```python
from urc.bridges.external_bridge import ExternalProcessBridge

bridge = ExternalProcessBridge(["ruta/a/tu-programa"])
print(bridge.reset())
print(bridge.observation_spec())
bridge.close()
```

Es mucho más fácil depurar un `reset()` suelto que un entrenamiento entero fallando a mitad.
