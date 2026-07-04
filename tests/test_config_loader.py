from pathlib import Path

import pytest

from urc.config import ConfigError
from urc.config.loader import (
    deep_merge,
    diff_configs,
    overrides_to_dict,
    parse_override,
    resolve_config,
)
from urc.config.schema import UrcConfig


def test_deep_merge_overrides_scalars_and_merges_nested_dicts():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"a": 2, "nested": {"y": 20, "z": 30}}

    merged = deep_merge(base, override)

    assert merged == {"a": 2, "nested": {"x": 1, "y": 20, "z": 30}}


def test_parse_override_interprets_value_as_yaml_scalar():
    assert parse_override("hyperparameters.learning_rate=3e-4") == (
        "hyperparameters.learning_rate",
        3e-4,
    )
    assert parse_override("bridge=socket") == ("bridge", "socket")
    assert parse_override("training.max_steps=1000") == ("training.max_steps", 1000)


def test_parse_override_without_equals_raises_config_error():
    with pytest.raises(ConfigError):
        parse_override("bridge")


def test_overrides_to_dict_builds_nested_structure_from_dotted_keys():
    result = overrides_to_dict(["hyperparameters.learning_rate=1e-4", "bridge=socket"])

    assert result == {"hyperparameters": {"learning_rate": 1e-4}, "bridge": "socket"}


def test_resolve_config_uses_library_defaults_when_nothing_else_given():
    config = resolve_config()

    # Los defaults empaquetados (defaults.yaml) traen hiperparámetros propios,
    # a diferencia de UrcConfig() "a pelo" (hyperparameters={}) — por eso no se
    # compara contra UrcConfig() sino contra el contenido esperado de ese YAML.
    assert config.bridge == "mlagents"
    assert config.algo == "sb3-ppo"
    assert config.hyperparameters == {
        "learning_rate": 3.0e-4,
        "gamma": 0.99,
    }


def test_resolve_config_layers_project_then_experiment_then_overrides(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("bridge: socket\nhyperparameters:\n  learning_rate: 0.001\n")

    experiment = tmp_path / "exp.yaml"
    experiment.write_text("hyperparameters:\n  gamma: 0.95\n")

    config = resolve_config(
        project_path=project,
        experiment_path=experiment,
        overrides={"hyperparameters": {"learning_rate": 0.0005}},
    )

    assert config.bridge == "socket"
    assert config.hyperparameters == {
        "learning_rate": 0.0005,
        "gamma": 0.95,
    }


def test_resolve_config_ignores_missing_project_file(tmp_path: Path):
    config = resolve_config(project_path=tmp_path / "does-not-exist.yaml")

    assert config == resolve_config()


def test_resolve_config_raises_config_error_with_readable_message_on_invalid_value(
    tmp_path: Path,
):
    project = tmp_path / "urc.yaml"
    project.write_text("logging:\n  backend: not-a-real-backend\n")

    with pytest.raises(ConfigError, match="logging.backend"):
        resolve_config(project_path=project)


def test_resolve_config_raises_config_error_on_non_mapping_yaml(tmp_path: Path):
    project = tmp_path / "urc.yaml"
    project.write_text("- just\n- a\n- list\n")

    with pytest.raises(ConfigError, match="mapeo"):
        resolve_config(project_path=project)


def test_diff_configs_reports_only_changed_leaf_paths():
    a = UrcConfig()
    b = UrcConfig(bridge="socket", hyperparameters={"learning_rate": 0.0005})

    differences = diff_configs(a, b)

    assert differences["bridge"] == ("mlagents", "socket")
    # diff_configs compara hoja a hoja, no dict a dict: el cambio en
    # hyperparameters se reporta en su clave anidada, no en el nivel superior.
    assert differences["hyperparameters.learning_rate"] == (None, 0.0005)
    assert "algo" not in differences
