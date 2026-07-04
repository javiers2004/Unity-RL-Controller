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
    """Bridge falso que SÍ soporta grabación (tiene set_time_scale/start_recording)."""

    def __init__(self) -> None:
        self.time_scale_calls: list[float] = []
        self.start_recording_calls: list[str] = []

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
    assert bridge.time_scale_calls == [50.0, 1.0]


def test_switches_back_to_fast_only_at_episode_boundary(tmp_path):
    bridge = RecordingBridge()
    callback = _make_callback(
        bridge, tmp_path, fast_forward_speed=50.0, normal_speed_every_n_steps=1000
    )
    callback._on_training_start()

    callback.num_timesteps = 1000
    callback.locals = {"dones": [False]}
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 1.0]  # entra en ventana normal

    callback.locals = {"dones": [False]}
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 1.0]  # sigue en la ventana, episodio no ha acabado

    callback.locals = {"dones": [True]}
    callback._on_step()
    assert bridge.time_scale_calls == [50.0, 1.0, 50.0]  # episodio termina, vuelve a rápido


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
