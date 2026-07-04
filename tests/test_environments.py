from urc.core.contracts import EnvironmentSpec
from urc.core.environments import register_environments_from_config, resolve_environment
from urc.core.registry import environments


def test_register_environments_from_config_builds_environment_spec():
    register_environments_from_config(
        {
            "test-env-a": {
                "build_path": "builds/a.exe",
                "bridge_options": {"no_graphics": True},
                "parameters": {"difficulty": 0.5},
                "curriculum": [{"parameters": {"difficulty": 0.1}, "min_reward": 1.0}],
            }
        }
    )

    spec = environments.get("test-env-a")

    assert spec == EnvironmentSpec(
        name="test-env-a",
        build_path="builds/a.exe",
        bridge_options={"no_graphics": True},
        parameters={"difficulty": 0.5},
        curriculum=[{"parameters": {"difficulty": 0.1}, "min_reward": 1.0}],
    )


def test_register_environments_from_config_is_idempotent_upsert():
    register_environments_from_config({"test-env-b": {"build_path": "old.exe"}})
    register_environments_from_config({"test-env-b": {"build_path": "new.exe"}})

    assert environments.get("test-env-b").build_path == "new.exe"


def test_resolve_environment_returns_registered_spec_when_declared():
    register_environments_from_config({"test-env-c": {"build_path": "c.exe"}})

    spec = resolve_environment("test-env-c", {"test-env-c": {"build_path": "c.exe"}})

    assert spec.name == "test-env-c"
    assert spec.build_path == "c.exe"


def test_resolve_environment_falls_back_to_bare_name_when_not_declared():
    spec = resolve_environment("free-form-label", {})

    assert spec == EnvironmentSpec(name="free-form-label")


def test_resolve_environment_defaults_to_default_name_when_env_is_none():
    spec = resolve_environment(None, {})

    assert spec == EnvironmentSpec(name="default")
