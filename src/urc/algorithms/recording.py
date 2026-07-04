from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from urc.core.contracts import BridgeAdapter


class RecordingCallback(BaseCallback):
    """Genera un vídeo del progreso del entrenamiento: la mayor parte a cámara
    rápida (`fast_forward_speed`), con ventanas a velocidad normal cada
    `normal_speed_every_n_steps` pasos, terminando con `final_episodes`
    episodios normales del agente ya entrenado.

    Solo funciona con bridges que expongan `set_time_scale`/`start_recording`
    (en la práctica, `MLAgentsBridge` + el script `UrcVideoRecorder.cs` en la
    escena — ver `unity/UrcVideoRecorder/`). Con cualquier otro bridge, avisa
    una vez y no hace nada más: el entrenamiento sigue igual, sin vídeo.

    La captura de fotogramas ocurre siempre en el lado de Unity mientras el
    script esté activo; aquí solo se controla `Time.timeScale` (que decide si
    los fotogramas capturados en ese tramo se ven rápidos o normales al
    reproducir el vídeo a un fps constante — ver ROADMAP, Fase 8, para el
    porqué de este diseño).
    """

    def __init__(
        self,
        bridge: BridgeAdapter,
        config: dict[str, Any],
        output_dir: Path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._bridge = bridge
        self._fast = float(config.get("fast_forward_speed", 20.0))
        self._every_n = int(config.get("normal_speed_every_n_steps", 1000))
        self._final_episodes = int(config.get("final_episodes", 4))
        self._fps = int(config.get("fps", 30))
        self._keep_frames = bool(config.get("keep_frames", False))
        # Absolutas a propósito: esta ruta se le manda tal cual a Unity (proceso
        # aparte, posiblemente con otro directorio de trabajo) para que cree ahí
        # los fotogramas — una ruta relativa se resolvería contra el cwd de
        # Unity, no el de `urc train`.
        output_dir = output_dir.resolve()
        self._frames_dir = output_dir / "video_frames"
        self._video_path = output_dir / "video" / "training_progress.mp4"
        self._supported = hasattr(bridge, "set_time_scale") and hasattr(bridge, "start_recording")
        self._next_normal_at = self._every_n
        self._in_normal_window = False

    def _on_training_start(self) -> None:
        if not self._supported:
            warnings.warn(
                "recording.enabled=true pero este bridge no soporta grabación de "
                "vídeo (de momento solo MLAgentsBridge, con UrcVideoRecorder.cs en "
                "la escena) — se ignora, el entrenamiento sigue sin vídeo.",
                stacklevel=2,
            )
            return
        self._bridge.start_recording(str(self._frames_dir))
        self._bridge.set_time_scale(self._fast)

    def _on_step(self) -> bool:
        if not self._supported:
            return True

        if not self._in_normal_window and self.num_timesteps >= self._next_normal_at:
            self._bridge.set_time_scale(1.0)
            self._in_normal_window = True

        dones = self.locals.get("dones")
        if self._in_normal_window and dones is not None and dones[0]:
            self._bridge.set_time_scale(self._fast)
            self._in_normal_window = False
            self._next_normal_at += self._every_n

        return True

    def _on_training_end(self) -> None:
        if not self._supported:
            return
        self._bridge.set_time_scale(1.0)
        self._record_final_episodes()
        self._assemble_video()

    def _record_final_episodes(self) -> None:
        for _ in range(self._final_episodes):
            obs = self._bridge.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                result = self._bridge.step(action)
                obs, done = result.observation, result.done

    def _assemble_video(self) -> None:
        import imageio.v2 as imageio

        frames = sorted(self._frames_dir.glob("frame_*.png"))
        if not frames:
            warnings.warn(
                "recording.enabled=true pero no se capturó ningún fotograma "
                "(¿está UrcVideoRecorder.cs en la escena?) — no se genera vídeo.",
                stacklevel=2,
            )
            return

        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(self._video_path), fps=self._fps) as writer:
            for frame_path in frames:
                writer.append_data(imageio.imread(frame_path))

        if not self._keep_frames:
            shutil.rmtree(self._frames_dir, ignore_errors=True)

        print(f"Vídeo de progreso guardado en: {self._video_path}")
