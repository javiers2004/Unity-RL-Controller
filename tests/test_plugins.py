import importlib.metadata
import sys

from urc.core.plugins import _plugin_module_name, load_entry_point_plugins, load_plugins_from_dir


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

    assert sys.modules[_plugin_module_name(plugin_file)].loaded is True


def test_load_plugins_from_dir_ignores_missing_directory(tmp_path):
    load_plugins_from_dir(tmp_path / "does-not-exist")


def test_load_plugins_from_dir_does_not_reimport_the_same_file_twice(tmp_path):
    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text("counter = counter + 1 if 'counter' in dir() else 1\n")

    load_plugins_from_dir(tmp_path)
    load_plugins_from_dir(tmp_path)

    assert sys.modules[_plugin_module_name(plugin_file)].counter == 1


def test_load_plugins_from_dir_gives_different_files_different_module_names(tmp_path):
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "same_name.py").write_text("marker = 'a'\n")
    (second_dir / "same_name.py").write_text("marker = 'b'\n")

    load_plugins_from_dir(first_dir)
    load_plugins_from_dir(second_dir)

    assert sys.modules[_plugin_module_name(first_dir / "same_name.py")].marker == "a"
    assert sys.modules[_plugin_module_name(second_dir / "same_name.py")].marker == "b"
