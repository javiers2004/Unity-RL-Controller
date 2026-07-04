import importlib.metadata
import sys

from urc.core.plugins import load_entry_point_plugins, load_plugins_from_dir


def test_load_entry_point_plugins_loads_each_registered_entry_point(monkeypatch):
    loaded = []

    class FakeEntryPoint:
        def load(self):
            loaded.append(True)

    def fake_entry_points(*, group):
        return [FakeEntryPoint()] if group == "urc.bridges" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

    load_entry_point_plugins()

    assert loaded == [True]


def test_load_plugins_from_dir_imports_each_python_file(tmp_path):
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text("loaded = True\n")

    load_plugins_from_dir(tmp_path)

    assert sys.modules["urc._plugins.my_plugin"].loaded is True


def test_load_plugins_from_dir_ignores_missing_directory(tmp_path):
    load_plugins_from_dir(tmp_path / "does-not-exist")
