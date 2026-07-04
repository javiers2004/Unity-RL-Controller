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
    assert config.logging.backend == "tensorboard"
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
