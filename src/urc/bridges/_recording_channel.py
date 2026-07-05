"""Canal de comunicación Python -> Unity para controlar la grabación de vídeo del
entrenamiento (ver `RecordingCallback` en `urc.algorithms.recording`).

Protocolo deliberadamente simple: un único string `comando|valor` por mensaje, sin
tipos estructurados — así el lado C# (`unity/UrcVideoRecorder/UrcVideoRecorder.cs`)
solo necesita un `Split('|')`. Python solo envía; no espera respuesta de Unity.
"""

from __future__ import annotations

import uuid

from mlagents_envs.side_channel.outgoing_message import OutgoingMessage
from mlagents_envs.side_channel.side_channel import IncomingMessage, SideChannel


class RecordingControlChannel(SideChannel):
    # Debe coincidir EXACTO con el GUID registrado en UrcVideoRecorder.cs.
    CHANNEL_ID = uuid.UUID("56274aad-fd18-43e7-8da5-f045b8ccea95")

    def __init__(self) -> None:
        super().__init__(self.CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass

    def start_recording(self, output_dir: str) -> None:
        self._send(f"start_recording|{output_dir}")

    def stop_recording(self) -> None:
        self._send("stop_recording|")

    def set_time_scale(self, scale: float) -> None:
        self._send(f"time_scale|{scale}")

    def _send(self, payload: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(payload)
        self.queue_message_to_send(msg)
