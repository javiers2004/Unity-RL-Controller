from pathlib import Path

from urc.core.runs import load_run_info, write_run_info


def test_write_and_load_run_info_round_trips(tmp_path: Path):
    checkpoint_dir = tmp_path / "runs" / "maze-v1"
    write_run_info(
        checkpoint_dir,
        bridge="socket",
        bridge_options={"host": "127.0.0.1", "port": 1234},
        algo="sb3-ppo",
        env="maze-v1",
    )

    checkpoint = checkpoint_dir / "checkpoint_100_steps.zip"
    info = load_run_info(checkpoint)

    assert info == {
        "bridge": "socket",
        "bridge_options": {"host": "127.0.0.1", "port": 1234},
        "algo": "sb3-ppo",
        "env": "maze-v1",
    }


def test_load_run_info_returns_empty_dict_when_missing(tmp_path: Path):
    checkpoint = tmp_path / "no_run_info_here" / "checkpoint.zip"

    assert load_run_info(checkpoint) == {}
