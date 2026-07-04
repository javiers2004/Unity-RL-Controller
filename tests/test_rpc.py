"""Tests de las piezas de bajo nivel de core/rpc.py que no necesitan un
proceso/socket real de por medio."""

import os

from urc.core.rpc import _resolve_executable_path


def test_resolve_executable_path_converts_relative_path_with_separator_to_absolute(
    tmp_path, monkeypatch
):
    # Bug real: en algunas instalaciones de Python en Windows (Microsoft
    # Store), subprocess.Popen con una ruta relativa con separador falla con
    # FileNotFoundError aunque el archivo exista de verdad.
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_executable_path(["sub/program.exe", "--flag"])

    assert resolved == [str(tmp_path / "sub" / "program.exe"), "--flag"]
    assert os.path.isabs(resolved[0])


def test_resolve_executable_path_leaves_bare_command_names_untouched():
    # Sin separador: debe resolverse por PATH, no por el directorio de trabajo.
    resolved = _resolve_executable_path(["python", "-c", "print(1)"])

    assert resolved == ["python", "-c", "print(1)"]


def test_resolve_executable_path_leaves_already_absolute_paths_untouched(tmp_path):
    absolute = str(tmp_path / "program.exe")

    assert _resolve_executable_path([absolute]) == [absolute]


def test_resolve_executable_path_handles_empty_command():
    assert _resolve_executable_path([]) == []
