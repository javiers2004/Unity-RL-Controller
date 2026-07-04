import pytest
from pydantic import ValidationError

from urc.config import UrcConfig


def test_default_config_has_sensible_values():
    config = UrcConfig()

    assert config.bridge == "mlagents"
    assert config.bridge_options == {}
    assert config.algo == "sb3-ppo"
    assert config.env is None
    assert config.hyperparameters == {}
    assert config.training.max_steps == 500_000
    assert config.training.checkpoint_every == 50_000
    assert config.training.progress_bar is False
    assert config.logging.backend == "tensorboard"
    assert config.logging.project == "urc"
    assert config.output_dir == "runs"


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValidationError):
        UrcConfig.model_validate({"brige": "mlagents"})


def test_unknown_logging_backend_is_rejected():
    with pytest.raises(ValidationError):
        UrcConfig.model_validate({"logging": {"backend": "not-a-real-backend"}})


def test_hyperparameters_accepts_arbitrary_keys():
    config = UrcConfig.model_validate({"hyperparameters": {"learning_rate": 1e-4, "custom": True}})

    assert config.hyperparameters == {"learning_rate": 1e-4, "custom": True}


def test_environments_section_parses_build_path_and_curriculum():
    config = UrcConfig.model_validate(
        {
            "environments": {
                "maze-v1": {
                    "build_path": "builds/maze.exe",
                    "bridge_options": {"no_graphics": True},
                    "curriculum": [
                        {"parameters": {"difficulty": 0.1}, "min_reward": 0.5},
                        {"parameters": {"difficulty": 0.9}},
                    ],
                }
            }
        }
    )

    env = config.environments["maze-v1"]
    assert env.build_path == "builds/maze.exe"
    assert env.bridge_options == {"no_graphics": True}
    assert env.curriculum[0].min_reward == 0.5
    assert env.curriculum[0].min_episodes == 1
    assert env.curriculum[1].min_reward is None


def test_environment_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        UrcConfig.model_validate({"environments": {"maze-v1": {"build_pth": "typo.exe"}}})


def test_lesson_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        UrcConfig.model_validate(
            {"environments": {"maze-v1": {"curriculum": [{"min_rewrd": 1.0}]}}}
        )
