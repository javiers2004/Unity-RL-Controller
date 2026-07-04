import sys
from pathlib import Path

from urc.bridges.external_bridge import ExternalProcessBridge

FIXTURE = Path(__file__).parent / "fixtures" / "echo_bridge.py"


def test_external_process_bridge_round_trips_over_stdio():
    bridge = ExternalProcessBridge([sys.executable, str(FIXTURE)])
    try:
        bridge.reset()

        obs_spec = bridge.observation_spec()
        assert obs_spec.shape == (2,)

        action_spec = bridge.action_spec()
        assert action_spec.shape == (1,)
        assert action_spec.discrete is False

        result = bridge.step(action=[0.5])
        assert result.reward == 1.0
        assert result.done is False
        assert result.observation == {"obs": [1.0, 1.0]}
    finally:
        bridge.close()
