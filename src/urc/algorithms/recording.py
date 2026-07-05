from __future__ import annotations

import bisect
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from urc.core.contracts import BridgeAdapter


class RecordingCallback(BaseCallback):
    """Genera un vídeo del progreso del entrenamiento: la mayor parte a cámara
    rápida (`fast_forward_speed`), con ventanas a cámara lenta
    (`normal_time_scale`) cada `normal_speed_every_n_steps` pasos para que se
    note la evolución, más ventanas extra a la misma velocidad que el final
    (`final_time_scale`, la más lenta) cada vez que la recompensa media de los
    últimos `stabilization_window` episodios alcanza un nuevo máximo — para
    poder ver el momento exacto de cada mejora real, no solo el resultado
    final. Termina con `final_episodes` episodios del agente ya entrenado.

    Cada fotograma del vídeo lleva quemado el paso de entrenamiento al que
    corresponde (o "Modelo final" durante los episodios finales).

    Solo funciona con bridges que expongan `set_time_scale`/`start_recording`
    (en la práctica, `MLAgentsBridge` + el script `UrcVideoRecorder.cs` en la
    escena — ver `unity/UrcVideoRecorder/`). Con cualquier otro bridge, avisa
    una vez y no hace nada más: el entrenamiento sigue igual, sin vídeo.

    La captura de fotogramas ocurre siempre en el lado de Unity mientras el
    script esté activo; aquí solo se controla `Time.timeScale` (que decide si
    los fotogramas capturados en ese tramo se ven rápidos o lentos al
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
        self._every_n = int(config.get("normal_speed_every_n_steps", 500))
        self._normal_time_scale = float(config.get("normal_time_scale", 0.25))
        self._final_episodes = int(config.get("final_episodes", 4))
        self._final_time_scale = float(config.get("final_time_scale", 0.05))
        self._stabilization_window = int(config.get("stabilization_window", 5))
        self._min_episodes_between_breakthroughs = int(
            config.get("min_episodes_between_breakthroughs", 20)
        )
        self._max_breakthroughs = int(config.get("max_breakthroughs", 5))
        self._fps = int(config.get("fps", 30))
        self._keep_frames = bool(config.get("keep_frames", False))
        # Absolutas a propósito: esta ruta se le manda tal cual a Unity (proceso
        # aparte, posiblemente con otro directorio de trabajo) para que cree ahí
        # los fotogramas — una ruta relativa se resolvería contra el cwd de
        # Unity, no el de `urc train`.
        output_dir = output_dir.resolve()
        self._frames_dir = output_dir / "video_frames"
        self._video_path = output_dir / "video" / "training_progress.mp4"
        self._supported = (
            hasattr(bridge, "set_time_scale")
            and hasattr(bridge, "start_recording")
            and hasattr(bridge, "stop_recording")
        )
        self._next_normal_at = self._every_n
        self._in_slow_window = False
        # Seguimiento de recompensa por episodio, para detectar mejoras reales
        # (ver _detected_breakthrough) — mismo patrón que CurriculumCallback.
        self._current_episode_reward = 0.0
        self._episode_rewards: list[float] = []
        self._best_avg_reward = float("-inf")
        self._episodes_since_last_breakthrough = 0
        self._breakthrough_count = 0
        # (tiempo real, num_timesteps) para poder etiquetar cada fotograma del
        # vídeo con el paso al que corresponde, a partir de la fecha de
        # modificación del PNG — ver _label_for_frame.
        self._timeline: list[tuple[float, int]] = []
        self._final_phase_start_time = float("inf")

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
        self._timeline.append((time.time(), 0))

    def _on_step(self) -> bool:
        if not self._supported:
            return True

        self._timeline.append((time.time(), self.num_timesteps))

        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._current_episode_reward += float(rewards[0])

        dones = self.locals.get("dones")
        episode_ended = bool(dones is not None and dones[0])
        if episode_ended:
            self._episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        if self._in_slow_window:
            if episode_ended:
                self._bridge.set_time_scale(self._fast)
                self._in_slow_window = False
            return True

        if episode_ended and self._detected_breakthrough():
            self._bridge.set_time_scale(self._final_time_scale)
            self._in_slow_window = True
        elif self.num_timesteps >= self._next_normal_at:
            self._bridge.set_time_scale(self._normal_time_scale)
            self._in_slow_window = True
            self._next_normal_at += self._every_n

        return True

    def _detected_breakthrough(self) -> bool:
        """Se dispara cuando la recompensa media de los últimos
        `stabilization_window` episodios marca un nuevo máximo, han pasado al
        menos `min_episodes_between_breakthroughs` episodios desde la última
        vez, y no se ha alcanzado ya `max_breakthroughs` en este entrenamiento.

        El tope máximo es necesario de verdad, no defensivo: en tareas que
        convergen rápido (verificado con Basic) la recompensa marca un nuevo
        máximo tan a menudo que, sin límite, la cámara lenta de cada
        "mejora" acababa dominando casi todo el vídeo en vez de ser un
        momento puntual — 4.017 fotogramas para solo 6.144 pasos entrenados,
        con más de mil fotogramas cubriendo un puñado de pasos.
        """
        self._episodes_since_last_breakthrough += 1
        if self._breakthrough_count >= self._max_breakthroughs:
            return False
        if len(self._episode_rewards) < self._stabilization_window:
            return False
        if self._episodes_since_last_breakthrough < self._min_episodes_between_breakthroughs:
            return False
        window = self._episode_rewards[-self._stabilization_window :]
        avg = sum(window) / len(window)
        if avg <= self._best_avg_reward:
            return False
        self._best_avg_reward = avg
        self._episodes_since_last_breakthrough = 0
        self._breakthrough_count += 1
        return True

    def _on_training_end(self) -> None:
        if not self._supported:
            return
        self._bridge.set_time_scale(self._final_time_scale)
        self._final_phase_start_time = time.time()
        self._record_final_episodes()
        # Antes de tocar la carpeta de fotogramas (ensamblar + borrar): la
        # captura en Unity es una corrutina en bucle infinito que, sin este
        # aviso, sigue intentando escribir ahí indefinidamente aunque ya
        # hayamos terminado — verificado contra Unity real (WallJump):
        # DirectoryNotFoundException en bucle tras borrar video_frames/.
        self._bridge.stop_recording()
        self._assemble_video()

    def _record_final_episodes(self) -> None:
        for _ in range(self._final_episodes):
            obs = self._bridge.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                result = self._bridge.step(action)
                obs, done = result.observation, result.done

    def _label_for_frame(self, mtime: float) -> str:
        if mtime >= self._final_phase_start_time:
            # Sin acento a propósito: la fuente de PIL.ImageFont.load_default()
            # no incluye caracteres acentuados (salía como un cuadrado en el
            # vídeo, verificado contra Unity real) — solo ASCII.
            return "Modelo final"
        if not self._timeline:
            return "Paso 0"
        times = [t for t, _ in self._timeline]
        index = min(bisect.bisect_left(times, mtime), len(self._timeline) - 1)
        step = self._timeline[index][1]
        return f"Paso {step:,}".replace(",", ".")

    def _assemble_video(self) -> None:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        frames = sorted(self._frames_dir.glob("frame_*.png"))
        if not frames:
            warnings.warn(
                "recording.enabled=true pero no se capturó ningún fotograma "
                "(¿está UrcVideoRecorder.cs en la escena?) — no se genera vídeo.",
                stacklevel=2,
            )
            return

        def draw_label(image: Image.Image, text: str, font: ImageFont.ImageFont) -> None:
            draw = ImageDraw.Draw(image)
            padding = 10
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = right - left, bottom - top
            box = (
                10,
                image.height - text_height - 2 * padding - 10,
                10 + text_width + 2 * padding,
                image.height - 10,
            )
            draw.rectangle(box, fill=(0, 0, 0))
            draw.text((box[0] + padding, box[1] + padding), text, font=font, fill=(255, 255, 255))

        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        font = None
        with imageio.get_writer(str(self._video_path), fps=self._fps) as writer:
            for frame_path in frames:
                image = Image.open(frame_path).convert("RGB")
                if font is None:
                    font = ImageFont.load_default(size=max(24, image.height // 30))
                label = self._label_for_frame(frame_path.stat().st_mtime)
                draw_label(image, label, font)
                writer.append_data(np.asarray(image))

        if not self._keep_frames:
            shutil.rmtree(self._frames_dir, ignore_errors=True)

        print(f"Vídeo de progreso guardado en: {self._video_path}")
