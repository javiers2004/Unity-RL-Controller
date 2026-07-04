# Protocolo out-of-process de `urc` (bridges en cualquier lenguaje)

> Especificación formal del protocolo que hablan `ExternalProcessBridge` (subproceso) y
> `SocketBridge` (TCP). Si quieres escribir el lado "entorno" de un bridge de `urc` en un lenguaje
> que no sea Python, este documento es el contrato completo — no necesitas leer el código de `urc`.
>
> **Alcance**: esto es solo para **bridges** (la conexión con el entorno de simulación). Los
> **algoritmos** (`AlgorithmBackend`) siguen siendo plugins de Python — ver la sección
> "Plugins de Python" del [ROADMAP](https://github.com/javiers2004/Unity-RL-Controller/blob/master/ROADMAP.md) (Fases 2 y 6). No hay (todavía) un protocolo
> out-of-process para algoritmos.

---

## 1. Idea general

Tu programa hace de "entorno": recibe acciones, devuelve observaciones/recompensas. `urc` es
siempre quien inicia la conversación — tu programa nunca llama a nada por iniciativa propia, solo
responde a lo que le llega.

Dos transportes posibles, mismo protocolo exacto por encima:

- **Subproceso** (`ExternalProcessBridge`): `urc` lanza tu programa y le habla por su
  `stdin`/`stdout`. Útil si tu entorno vive en el mismo lenguaje/proceso que quieres empaquetar
  junto a `urc` (p. ej. un ejecutable compilado).
- **Socket TCP** (`SocketBridge`): `urc` se conecta a un `host:port` donde ya tienes un servidor
  escuchando. Útil si tu entorno corre por separado (otra máquina, un proceso de larga duración,
  Unity con un socket propio, etc.).

En ambos casos: **una petición JSON por línea, una respuesta JSON por línea**, en ese orden
(petición-respuesta, nunca dos peticiones seguidas sin respuesta de por medio).

## 2. Formato de las líneas

Cada línea es exactamente un objeto JSON, codificado en UTF-8 y terminado en `\n` (no hace falta
`\r\n`; si tu lenguaje solo sabe escribir el fin de línea nativo del sistema operativo, no pasa
nada — el lado Python normaliza cualquier estilo de fin de línea al leer).

**Petición** (la manda siempre `urc`):

```json
{"method": "step", "params": {"action": [0.1]}}
```

- `method`: string, uno de los de la sección 3.
- `params`: el objeto de parámetros del método, o `null`/ausente si el método no necesita ninguno.

**Respuesta** (la manda siempre tu programa):

```json
{"result": {"observation": [0.0], "reward": 1.0, "done": false, "info": {}}}
```

o, si algo fue mal:

```json
{"error": "mensaje legible describiendo qué pasó"}
```

Una línea de respuesta lleva **una de las dos claves, nunca ambas**. Si tu respuesta trae
`"error"`, `urc` lanza una excepción (`RpcError`) con ese mensaje tal cual — que sea legible para
un humano depurando, no un código de error.

## 3. Métodos

| Método | `params` | `result` en éxito |
|---|---|---|
| `reset` | (ninguno) | la observación inicial, con la forma declarada por `observation_spec` |
| `step` | `{"action": <acción>}` | `{"observation": ..., "reward": <número>, "done": <bool>, "info": <objeto, opcional>}` |
| `observation_spec` | (ninguno) | `{"shape": [<int>, ...], "dtype": <string, opcional, default "float32">}` |
| `action_spec` | (ninguno) | `{"shape": [<int>, ...], "dtype": <string, opcional>, "discrete": <bool, opcional, default false>, "discrete_branches": [<int>, ...] (solo si discrete=true, uno por rama)}` |
| `set_parameters` | `{"parameters": {<clave>: <número>, ...}}` | cualquier cosa (se ignora) — ver nota abajo |

Notas:

- `action` en `step` tiene la forma que describiste en `action_spec`: un array de `shape[0]`
  números para acciones continuas, o un entero (una rama) / array de enteros (varias ramas) para
  acciones discretas.
- `info` en la respuesta de `step` es opcional; si lo omites, `urc` lo trata como `{}`. Si tu
  entorno sabe si el episodio fue un "éxito" o no, mete `"success": true/false` ahí — `urc eval`
  lo usa automáticamente si está presente (ver ROADMAP, Fase 8).
- Con acciones discretas, `shape` es el **número de ramas**, no el número de valores posibles por
  rama — eso es lo que dice `discrete_branches` (uno por rama). Para una acción discreta con 3
  valores posibles (una sola rama): `"shape": [1], "discrete": true, "discrete_branches": [3]`.
- `set_parameters` es **opcional de implementar de verdad**: si no vas a usar currículo ni domain
  randomization, responde igualmente con éxito (p. ej. `{"result": null}`) en vez de con un
  `"error"` — así el resto del contrato (`urc train` con un `curriculum` declarado) no se rompe
  solo porque tu entorno no soporta esa característica. Devolver un error ahí se trata como un
  fallo real, no como "no soportado".
- No existe un método `close`: cerrar la conexión (fin de `stdin` en subprocesos, cierre del
  socket en TCP) es la señal de que `urc` ha terminado. Tu programa debería salir en cuanto
  detecte el cierre (en la mayoría de lenguajes, leer una línea y recibir "fin de fichero"/`null`
  es esa señal). `urc` no espera mucho a que salgas solo: si sigues vivo justo después de cerrar
  la entrada, te fuerza a terminar.

## 4. Un cuidado con los números

JSON no distingue "entero" de "flotante" a nivel de sintaxis — eso lo decide el serializador de
cada lenguaje. Por ejemplo, el bridge de referencia en C# (sección 5) serializa `0.0` como `0`
literalmente, y al otro lado Python lo interpreta como `int`. Esto no es un bug: `0 == 0.0` en
Python, y `urc` no asume un tipo concreto en los números que le llegan. Si escribes tu propio
bridge, no necesitas preocuparte por forzar floats "con pinta de float" — cualquier representación
numérica válida de JSON sirve.

## 5. Implementación de referencia: C#

[`examples/csharp_bridge/Program.cs`](https://github.com/javiers2004/Unity-RL-Controller/blob/master/examples/csharp_bridge/Program.cs) es un bridge completo en
C#, sin ninguna dependencia de `urc` ni de Python — solo usa `JavaScriptSerializer`
(`System.Web.Extensions`, incluido en .NET Framework, sin necesitar NuGet). Implementa un entorno
de juguete (episodios de 3 pasos, recompensa fija) para poder probarlo sin nada más.

Compilar en Windows sin instalar el SDK de .NET (usa el compilador que ya trae .NET Framework):

```powershell
C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe `
  /reference:System.Web.Extensions.dll /out:bridge.exe examples\csharp_bridge\Program.cs
```

Probarlo desde `urc` (subproceso):

```yaml
bridge: external
bridge_options:
  command: ["ruta/a/bridge.exe"]
```

O directamente desde Python:

```python
from urc.bridges.external_bridge import ExternalProcessBridge

bridge = ExternalProcessBridge(["ruta/a/bridge.exe"])
bridge.reset()
```

`tests/test_csharp_reference_bridge.py` compila este mismo archivo y lo conecta con un
`ExternalProcessBridge` real en cada ejecución del test (se salta solo si no hay `csc.exe`
disponible, p. ej. en un runner Linux) — no es solo un ejemplo documentado, es una prueba de
interoperabilidad real y automatizada.

Para un ejemplo más simple (menos código, pero en Python) mirando exactamente el mismo protocolo,
[`tests/fixtures/echo_bridge.py`](https://github.com/javiers2004/Unity-RL-Controller/blob/master/tests/fixtures/echo_bridge.py) hace lo mismo en ~30 líneas.

## 6. Checklist para escribir tu propio bridge

1. Elige transporte: subproceso (lee `stdin`, escribe `stdout`) o servidor TCP.
2. Por cada línea de entrada, parsea el JSON y mira `method`.
3. Implementa como mínimo `reset`, `step`, `observation_spec`, `action_spec`. `set_parameters` es
   opcional (pero responde con éxito igualmente, ver sección 3).
4. Cada respuesta es una línea JSON con `"result"` o `"error"`, terminada en salto de línea, con
   flush inmediato (no dejes la respuesta en un buffer sin vaciar: `urc` se queda esperando).
5. Prueba tu bridge conectándolo directamente con `ExternalProcessBridge`/`SocketBridge` antes de
   usarlo desde `urc train` — es mucho más fácil depurar un `bridge.reset()` suelto que un
   entrenamiento entero.
6. Cuando funcione, decláralo en tu proyecto (`bridge: external` o `bridge: socket` en
   `urc.yaml`, con los `bridge_options` que necesite tu programa) y ya puedes usar `urc train`,
   `urc eval` y `urc record` con él, igual que con el bridge de ML-Agents (todos resuelven el
   bridge por nombre desde la config). `urc env launch` es la excepción: hoy solo sabe hablar con
   `mlagents` directamente, no con bridges arbitrarios — usa `bridge.reset()` a mano (paso 5) como
   sustituto para probar la conexión con tu propio bridge.
