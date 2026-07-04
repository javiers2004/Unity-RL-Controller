// Bridge de referencia en C# para el protocolo out-of-process de urc.
//
// Demuestra que cualquier lenguaje puede implementar el lado "entorno" del
// contrato BridgeAdapter sin ninguna librería de urc ni de Python: solo hace
// falta leer/escribir líneas JSON por stdin/stdout. Ver PROTOCOL.md en la
// raíz del repo para la especificación completa.
//
// Compilar (Windows, sin instalar el SDK de .NET, con el compilador que ya
// trae .NET Framework):
//   C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe ^
//     /reference:System.Web.Extensions.dll /out:bridge.exe Program.cs
//
// Probarlo desde Python:
//   from urc.bridges.external_bridge import ExternalProcessBridge
//   bridge = ExternalProcessBridge(["bridge.exe"])
//   bridge.reset()
//
// O desde urc.yaml: bridge: external, bridge_options: { command: ["bridge.exe"] }

using System;
using System.Collections.Generic;
using System.Web.Script.Serialization;

internal static class Program
{
    private static void Main()
    {
        var json = new JavaScriptSerializer();
        var episodeStep = 0;
        const int episodeLength = 3;

        string line;
        while ((line = Console.In.ReadLine()) != null)
        {
            if (line.Length == 0)
            {
                continue;
            }

            var request = (Dictionary<string, object>)json.DeserializeObject(line);
            var method = (string)request["method"];
            object result;

            switch (method)
            {
                case "reset":
                    episodeStep = 0;
                    result = new object[] { 0.0 };
                    break;

                case "step":
                    episodeStep++;
                    var done = episodeStep >= episodeLength;
                    if (done)
                    {
                        episodeStep = 0;
                    }

                    result = new Dictionary<string, object>
                    {
                        { "observation", new object[] { 0.0 } },
                        { "reward", 1.0 },
                        { "done", done },
                        { "info", new Dictionary<string, object>() },
                    };
                    break;

                case "observation_spec":
                    result = new Dictionary<string, object>
                    {
                        { "shape", new object[] { 1 } },
                        { "dtype", "float32" },
                    };
                    break;

                case "action_spec":
                    result = new Dictionary<string, object>
                    {
                        { "shape", new object[] { 1 } },
                        { "dtype", "float32" },
                        { "discrete", false },
                    };
                    break;

                case "set_parameters":
                    // Un bridge real aplicaría aquí los parámetros recibidos
                    // (domain randomization/currículo) al entorno de verdad.
                    result = null;
                    break;

                default:
                    WriteLine(json, new Dictionary<string, object> { { "error", "método desconocido: " + method } });
                    continue;
            }

            WriteLine(json, new Dictionary<string, object> { { "result", result } });
        }
    }

    private static void WriteLine(JavaScriptSerializer json, Dictionary<string, object> payload)
    {
        Console.Out.Write(json.Serialize(payload));
        Console.Out.Write('\n');
        Console.Out.Flush();
    }
}
