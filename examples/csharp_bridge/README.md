# Ejemplo: bridge en C# (SAC)

Bridge de referencia en C# ([`Program.cs`](Program.cs)) para demostrar que el protocolo de `urc`
funciona en cualquier lenguaje — ver la especificación completa en
[PROTOCOL.md](../../PROTOCOL.md). No usa Unity ni ninguna librería de `urc`: es un programa de
consola independiente que habla JSON por `stdin`/`stdout`.

Este ejemplo entrena con **SAC** en vez de PPO (la tarea de juguete tiene acciones continuas, así
que sirve para probar el segundo algoritmo — ver Fase 6 del [ROADMAP](../../ROADMAP.md)).

## Compilar

En Windows, sin instalar el SDK de .NET (usa el compilador que ya trae .NET Framework):

```powershell
C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe `
  /reference:System.Web.Extensions.dll /out:bridge.exe Program.cs
```

(En Linux/macOS, compílalo con Mono: `mcs -r:System.Web.Extensions.dll -out:bridge.exe Program.cs`
y ejecútalo con `mono bridge.exe` — en ese caso, cambia `command` en `urc.yaml` a
`["mono", "examples/csharp_bridge/bridge.exe"]`.)

## Probarlo

```bash
cd examples/csharp_bridge
urc train      # usa este urc.yaml: bridge external, algoritmo sb3-sac
urc eval runs/default/checkpoint_2000_steps.zip --episodes 10
```

`urc train` lanza `bridge.exe` como subproceso él solo (no hace falta arrancar nada en otra
terminal, a diferencia del ejemplo `toy_reach_target`) y habla con él por `stdin`/`stdout`.

## Nota técnica

Este ejemplo reveló un bug real de Windows durante el desarrollo: en algunas instalaciones de
Python (p. ej. la de Microsoft Store), `subprocess.Popen` con una ruta relativa que contiene
separadores de carpeta fallaba con `FileNotFoundError` aunque el archivo existiera de verdad —
`CreateProcess` no la resolvía contra el directorio de trabajo igual que las APIs de archivo
normales de Python. `urc` ahora convierte esas rutas a absolutas automáticamente antes de lanzar
el proceso (ver `_resolve_executable_path` en `urc.core.rpc`), así que `command` en `bridge_options`
puede ser una ruta relativa sin problema, sea cual sea la instalación de Python del usuario.
