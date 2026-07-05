using System;
using System.Collections;
using System.Globalization;
using System.IO;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace Urc
{
    /// <summary>
    /// Adjuntar a cualquier GameObject de la escena para que `urc train
    /// --set recording.enabled=true` pueda generar un vídeo del progreso del
    /// entrenamiento (ver RecordingCallback en el repo de urc, y
    /// examples/unity_basic_ppo/README.md para las instrucciones de uso).
    ///
    /// Captura fotogramas PNG (a un ritmo real fijo, ver CaptureIntervalSeconds)
    /// en la carpeta que indique Python, y expone `Time.timeScale` a través de
    /// un side channel — Python controla el ritmo (cámara rápida/normal) solo
    /// con eso; la captura en sí nunca se pausa una vez empezada.
    /// </summary>
    public class UrcVideoRecorder : MonoBehaviour
    {
        // No se usa ScreenCapture.CaptureScreenshot(path): encola la captura de
        // forma asíncrona y solo admite una "en vuelo" a la vez — en pruebas
        // reales, con llamadas repetidas durante varios minutos, solo 1-2 de
        // miles de peticiones llegaban a completarse (el resto se descartaban en
        // silencio, sin ningún error en consola). En su lugar, se lee el
        // framebuffer directamente con Texture2D.ReadPixels dentro de una
        // corrutina con WaitForEndOfFrame — técnica síncrona y fiable, sin cola
        // interna que pueda atascarse.
        private const float CaptureIntervalSeconds = 0.1f;

        private class Channel : SideChannel
        {
            // Debe coincidir EXACTO con RecordingControlChannel.CHANNEL_ID en
            // src/urc/bridges/_recording_channel.py.
            private const string ChannelGuid = "56274aad-fd18-43e7-8da5-f045b8ccea95";

            public event Action<string> OnStartRecording;
            public event Action OnStopRecording;
            public event Action<float> OnSetTimeScale;

            public Channel()
            {
                ChannelId = new Guid(ChannelGuid);
            }

            protected override void OnMessageReceived(IncomingMessage msg)
            {
                var payload = msg.ReadString();
                var parts = payload.Split(new[] { '|' }, 2);
                if (parts.Length != 2)
                {
                    return;
                }

                switch (parts[0])
                {
                    case "start_recording":
                        OnStartRecording?.Invoke(parts[1]);
                        break;
                    case "stop_recording":
                        OnStopRecording?.Invoke();
                        break;
                    case "time_scale":
                        // Cultura invariante a propósito: Python siempre manda "." como
                        // separador decimal (str(float) en Python usa el formato "C"), pero
                        // float.TryParse(string) sin cultura usa la del sistema operativo —
                        // en español (",", "." de miles) "50.0" se leería como 500.
                        if (float.TryParse(
                            parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var scale))
                        {
                            OnSetTimeScale?.Invoke(scale);
                        }
                        break;
                }
            }
        }

        // Estáticas a propósito: muchas escenas de ejemplo de ML-Agents (Basic
        // incluida) resetean el episodio recargando la escena ENTERA
        // (SceneManager.LoadScene), lo que destruye y recrea este componente en
        // cada episodio. Con campos de instancia normales, cada recarga perdía
        // la carpeta de salida y reiniciaba el contador a 0 (sobrescribiendo
        // fotogramas ya guardados) — con estáticos, sobreviven a la recarga
        // como si nada (Time.timeScale ya es global de por sí y no hacía falta
        // este tratamiento, pero sí se veía afectado indirectamente: ver
        // OnDisable más abajo).
        private static string s_OutputDir;
        private static int s_FrameCounter;
        private static float s_LastCaptureTime = float.NegativeInfinity;
        private static Texture2D s_CaptureTexture;

        private Channel _channel;

        private void OnEnable()
        {
            _channel = new Channel();
            _channel.OnStartRecording += HandleStartRecording;
            _channel.OnStopRecording += HandleStopRecording;
            _channel.OnSetTimeScale += HandleSetTimeScale;
            SideChannelManager.RegisterSideChannel(_channel);
            StartCoroutine(CaptureLoop());
        }

        private void OnDisable()
        {
            // No se resetea Time.timeScale aquí: OnDisable también se dispara
            // en cada recarga de escena (ver nota de los campos estáticos
            // arriba), y resetearlo ahí deshacía el "modo rápido" en cada
            // episodio — justo lo que RecordingCallback controla desde Python.
            StopAllCoroutines();
            if (_channel == null)
            {
                return;
            }
            SideChannelManager.UnregisterSideChannel(_channel);
            _channel.OnStartRecording -= HandleStartRecording;
            _channel.OnStopRecording -= HandleStopRecording;
            _channel.OnSetTimeScale -= HandleSetTimeScale;
            _channel = null;
        }

        private void HandleStartRecording(string outputDir)
        {
            s_OutputDir = outputDir;
            s_FrameCounter = 0;
            Directory.CreateDirectory(s_OutputDir);
        }

        private void HandleStopRecording()
        {
            // Sin esto, la corrutina de captura (bucle infinito) sigue
            // intentando escribir en s_OutputDir indefinidamente, aunque
            // Python ya haya terminado y borrado esa carpeta al ensamblar el
            // vídeo — verificado contra Unity real (WallJump):
            // DirectoryNotFoundException en bucle tras el borrado.
            s_OutputDir = null;
        }

        private void HandleSetTimeScale(float scale)
        {
            Time.timeScale = scale;
        }

        private IEnumerator CaptureLoop()
        {
            while (true)
            {
                yield return new WaitForEndOfFrame();
                if (string.IsNullOrEmpty(s_OutputDir))
                {
                    continue;
                }
                if (Time.unscaledTime - s_LastCaptureTime < CaptureIntervalSeconds)
                {
                    continue;
                }
                s_LastCaptureTime = Time.unscaledTime;
                CaptureFrame();
            }
        }

        private void CaptureFrame()
        {
            int width = Screen.width;
            int height = Screen.height;
            if (s_CaptureTexture == null || s_CaptureTexture.width != width || s_CaptureTexture.height != height)
            {
                s_CaptureTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
            }
            s_CaptureTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            s_CaptureTexture.Apply();

            var path = Path.Combine(s_OutputDir, $"frame_{s_FrameCounter:D6}.png");
            try
            {
                File.WriteAllBytes(path, s_CaptureTexture.EncodeToPNG());
            }
            catch (DirectoryNotFoundException)
            {
                // El mensaje "stop_recording" de Python se envía en el siguiente
                // intercambio del comunicador, que puede no llegar antes de que
                // Python ya haya ensamblado el vídeo y borrado esta carpeta —
                // condición de carrera real, verificada contra Unity (WallJump).
                // Se ignora en vez de dejar una excepción sin capturar: para
                // entonces el vídeo ya está guardado, este fotograma ya no sirve.
                s_OutputDir = null;
                return;
            }
            s_FrameCounter++;
        }
    }
}
