"""Integración real contra un build headless de Unity (escena Basic de
ML-Agents). No forma parte del `pytest` normal de cada commit — se salta si
no hay un build disponible. Solo corre en el workflow `unity-integration.yml`
de CI (descarga el build de un GitHub Release, ver ROADMAP.md Fase 11), o en
local si exportas tu propio build headless y apuntas `URC_UNITY_BUILD_PATH`
a su ejecutable.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("mlagents_envs")

BUILD_PATH = os.environ.get("URC_UNITY_BUILD_PATH")

pytestmark = pytest.mark.skipif(
    not BUILD_PATH or not Path(BUILD_PATH).exists(),
    reason="URC_UNITY_BUILD_PATH no apunta a un build headless real de Unity.",
)


def test_mlagents_bridge_connects_to_real_headless_unity_build():
    from urc.bridges.mlagents_bridge import MLAgentsBridge

    bridge = MLAgentsBridge(file_name=BUILD_PATH, no_graphics=True, worker_id=0, timeout_wait=120)
    try:
        obs = bridge.reset()
        assert len(obs) == 20  # observación one-hot de 20 posiciones de la escena Basic

        obs_spec = bridge.observation_spec()
        assert obs_spec.shape == (20,)

        action_spec = bridge.action_spec()
        assert action_spec.discrete is True
        assert action_spec.shape == (1,)
        assert action_spec.discrete_branches == (3,)

        result = bridge.step(action=1)
        assert isinstance(result.reward, float)
    finally:
        bridge.close()
