"""Tests de RecordingCallback contra un bridge falso y el bucle de SB3 simulado
a mano (mismo patrón que tests/test_curriculum.py): no hace falta entrenar de
verdad ni tener Unity, solo stable-baselines3 instalado (por `BaseCallback`).
"""

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from urc.algorithms.recording import RecordingCallback  # noqa: E402
from urc.core.contracts import BridgeAdapter  # noqa: E402


class RecordingBridge(BridgeAdapter):
    """Bridge falso que SÍ soporta grabación (set_time_scale/start_recording/stop_recording)."""

    def __init__(self) -> None:
        self.time_scale_calls: list[float] = []
        self.start_recording_calls: list[str] = []
        self.stop_recording_calls: int = 0

    def reset(self):
        return "obs"

    def step(self, action):
        raise NotImplementedError

    def observation_spec(self):
        raise NotImplementedError

    def action_spec(self):
        raise NotImplementedError

    def close(self):
        pass

    def set_time_scale(self, scale: float) -> None:
        self.time_scale_calls.append(scale)

    def start_recording(self, output_dir: str) -> None:
        self.start_recording_calls.append(output_dir)

    def stop_recording(self) -> None:
        self.stop_recording_calls += 1


class PlainBridge(BridgeAdapter):
    """Bridge falso sin soporte de grabación (no define set_time_scale/start_recording)."""

    def reset(self):
        return "obs"

    def step(self, action):
        raise NotImplementedError

    def observation_spec(self):
        raise NotImplementedError

    def action_spec(self):
        raise NotImplementedError

    def close(self):
        pass


def _make_callback(bridge: BridgeAdapter, output_dir: Path, **config) -> RecordingCallback:
    callback = RecordingCallback(bridge, config, output_dir)
    callback.locals = {}
    callback.num_timesteps = 0
    return callback


def test_on_training_start_begins_recording_at_fast_speed(tmp_path):
    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path, fast_forward_speed=50.0)

    callback._on_training_start()

    assert bridge.start_recording_calls == [str(tmp_path / "video_frames")]
    assert bridge.time_scale_calls == [50.0]


def test_switches_to_normal_speed_after_crossing_threshold(tmp_path):
    bridge = RecordingBridge()
    callback = _make_callback(
        bridge, tmp_path, fast_forward_speed=50.0, normal_speed_every_n_steps=1000
    )
    callback._on_training_start()

    callback.num_timesteps = 999
    callback._on_step()
    assert bridge.time_scale_calls == [50.0]  # todavía no ha cruzado el umbral

    callback.num_timesteps = 1000
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 0.25]  # normal_time_scale por defecto


def test_switches_back_to_fast_only_at_episode_boundary(tmp_path):
    bridge = RecordingBridge()
    callback = _make_callback(
        bridge, tmp_path, fast_forward_speed=50.0, normal_speed_every_n_steps=1000
    )
    callback._on_training_start()

    callback.num_timesteps = 1000
    callback.locals = {"dones": [False]}
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 0.25]  # entra en ventana lenta

    callback.locals = {"dones": [False]}
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 0.25]  # sigue en la ventana, episodio no ha acabado

    callback.locals = {"dones": [True]}
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 0.25, 50.0]  # episodio termina, vuelve a rápido


def test_records_breakthrough_on_first_stable_window_and_on_further_improvement(tmp_path):
    """Traza verificada a mano contra la implementación real (ver commit): la
    primera ventana estable posible siempre dispara un clip (primera
    evidencia de comportamiento consistente); mientras la recompensa se
    mantiene plana no vuelve a disparar; un nuevo máximo sí, respetando el
    margen mínimo de episodios entre disparos."""
    bridge = RecordingBridge()
    callback = _make_callback(
        bridge,
        tmp_path,
        fast_forward_speed=50.0,
        stabilization_window=3,
        min_episodes_between_breakthroughs=3,
    )
    callback._on_training_start()

    def end_episode(reward: float) -> None:
        callback.locals = {"rewards": [reward], "dones": [True]}
        callback._on_step()

    for _ in range(2):
        end_episode(1.0)
    assert bridge.time_scale_calls == [50.0]  # todavía sin 3 episodios registrados

    end_episode(1.0)  # 3er episodio flojo: primera ventana estable posible -> dispara
    assert bridge.time_scale_calls == [50.0, 0.05]

    end_episode(1.0)  # episodio siguiente: la ventana lenta se cierra al terminar
    assert bridge.time_scale_calls == [50.0, 0.05, 50.0]

    for _ in range(2):
        end_episode(1.0)  # mismo nivel que antes: no es un nuevo máximo
    assert bridge.time_scale_calls == [50.0, 0.05, 50.0]

    end_episode(5.0)  # la media de los últimos 3 sube a un nuevo máximo -> dispara otra vez
    assert bridge.time_scale_calls == [50.0, 0.05, 50.0, 0.05]


def test_max_breakthroughs_caps_slow_motion_even_if_reward_keeps_improving(tmp_path):
    """Sin este tope, en tareas que convergen rápido (Basic, verificado contra
    Unity real) la recompensa marca un nuevo máximo tan a menudo que la cámara
    lenta domina casi todo el vídeo — ver ROADMAP, Fase 8, 2026-07-05."""
    bridge = RecordingBridge()
    callback = _make_callback(
        bridge,
        tmp_path,
        fast_forward_speed=50.0,
        stabilization_window=1,
        min_episodes_between_breakthroughs=1,
        max_breakthroughs=1,
    )
    callback._on_training_start()

    def end_episode(reward: float) -> None:
        callback.locals = {"rewards": [reward], "dones": [True]}
        callback._on_step()

    end_episode(1.0)  # primer episodio: dispara el único breakthrough permitido
    assert bridge.time_scale_calls == [50.0, 0.05]
    end_episode(1.0)  # cierra la ventana lenta
    assert bridge.time_scale_calls == [50.0, 0.05, 50.0]

    end_episode(10.0)  # nuevo máximo claro, pero ya se alcanzó max_breakthroughs=1
    assert bridge.time_scale_calls == [50.0, 0.05, 50.0]
    end_episode(20.0)  # otro nuevo máximo: tampoco dispara
    assert bridge.time_scale_calls == [50.0, 0.05, 50.0]


def test_on_training_end_stops_recording_before_assembling_video(tmp_path):
    """stop_recording debe llamarse antes de tocar la carpeta de fotogramas:
    sin esto, la corrutina de captura en Unity sigue escribiendo ahí después
    de que el vídeo ya se haya ensamblado y la carpeta se haya borrado —
    verificado contra Unity real (WallJump), ver ROADMAP Fase 8."""
    pytest.importorskip("imageio")

    class FakeModel:
        def predict(self, observation, deterministic=True):
            return "action", None

    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path, final_episodes=0)
    callback.model = FakeModel()

    with pytest.warns(UserWarning, match="no se capturó ningún fotograma"):
        callback._on_training_end()  # sin fotogramas de verdad en este test

    assert bridge.stop_recording_calls == 1


def test_unsupported_bridge_does_nothing_and_does_not_raise(tmp_path):
    bridge = PlainBridge()
    callback = _make_callback(bridge, tmp_path)

    with pytest.warns(UserWarning, match="no soporta grabación"):
        callback._on_training_start()

    callback.num_timesteps = 10_000
    callback._on_step()  # no debe lanzar excepción aunque el bridge no tenga los métodos
    callback._on_training_end()  # idem


def test_assemble_video_writes_mp4_from_captured_frames(tmp_path):
    imageio = pytest.importorskip("imageio")
    import numpy as np

    frames_dir = tmp_path / "video_frames"
    frames_dir.mkdir()
    for i in range(3):
        frame = np.full((8, 8, 3), fill_value=i * 10, dtype=np.uint8)
        imageio.imwrite(frames_dir / f"frame_{i:06d}.png", frame)

    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path, fps=10)

    callback._assemble_video()

    video_path = tmp_path / "video" / "training_progress.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0
    assert not frames_dir.exists()  # keep_frames=False por defecto


def test_assemble_video_keeps_frames_when_requested(tmp_path):
    imageio = pytest.importorskip("imageio")
    import numpy as np

    frames_dir = tmp_path / "video_frames"
    frames_dir.mkdir()
    imageio.imwrite(frames_dir / "frame_000000.png", np.zeros((8, 8, 3), dtype=np.uint8))

    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path, keep_frames=True)

    callback._assemble_video()

    assert frames_dir.exists()


def test_assemble_video_warns_when_no_frames_captured(tmp_path):
    pytest.importorskip("imageio")
    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path)

    with pytest.warns(UserWarning, match="no se capturó ningún fotograma"):
        callback._assemble_video()

    assert not (tmp_path / "video" / "training_progress.mp4").exists()


def test_label_for_frame_interpolates_step_from_timeline(tmp_path):
    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path)
    callback._timeline = [(100.0, 0), (110.0, 5_000), (120.0, 10_000)]

    assert callback._label_for_frame(100.0) == "Paso 0"
    assert callback._label_for_frame(110.0) == "Paso 5.000"
    assert callback._label_for_frame(115.0) == "Paso 10.000"  # entre puntos: el siguiente conocido
    assert callback._label_for_frame(200.0) == "Paso 10.000"  # más allá del último punto


def test_label_for_frame_returns_final_label_after_final_phase(tmp_path):
    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path)
    callback._timeline = [(100.0, 0), (110.0, 5_000)]
    callback._final_phase_start_time = 150.0

    assert callback._label_for_frame(110.0) == "Paso 5.000"
    assert callback._label_for_frame(150.0) == "Modelo final"
    assert callback._label_for_frame(200.0) == "Modelo final"


def test_assemble_video_burns_step_label_into_frames(tmp_path):
    imageio = pytest.importorskip("imageio")
    pytest.importorskip("PIL")
    import numpy as np

    frames_dir = tmp_path / "video_frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "frame_000000.png"
    imageio.imwrite(frame_path, np.zeros((200, 200, 3), dtype=np.uint8))

    bridge = RecordingBridge()
    callback = _make_callback(bridge, tmp_path)
    callback._final_phase_start_time = 0.0  # cualquier fotograma cuenta como "final"

    callback._assemble_video()

    video_path = tmp_path / "video" / "training_progress.mp4"
    reader = imageio.get_reader(str(video_path))
    frame = reader.get_data(0)
    reader.close()
    # La etiqueta se dibuja como texto blanco sobre una caja negra en la
    # esquina inferior izquierda — si se dibujó, esa zona ya no es todo ceros.
    assert frame[-20:, :150].max() > 0
