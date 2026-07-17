from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/setup_cml.py"
_SPEC = importlib.util.spec_from_file_location("setup_cml", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
setup_cml = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(setup_cml)


@pytest.fixture
def cml_paths(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    requirements = project / "requirements-cml.txt"
    requirements.write_text("package==1\n", encoding="utf-8")
    environment = project / ".cml-venv"
    environment_root = project / ".cml-venvs"
    monkeypatch.setattr(setup_cml, "ROOT", project)
    monkeypatch.setattr(setup_cml, "ENVIRONMENT", environment)
    monkeypatch.setattr(setup_cml, "ENVIRONMENT_ROOT", environment_root)
    monkeypatch.setattr(setup_cml, "REQUIREMENTS", requirements)
    return project, environment, environment_root


def _successful_subprocess(monkeypatch, site_packages: Path) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        del kwargs
        command = [str(part) for part in command]
        calls.append(command)
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("python", encoding="utf-8")
        elif "site.getsitepackages" in command[-1]:
            site_packages.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(stdout=f"{site_packages}\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(setup_cml.subprocess, "run", run)
    return calls


def test_build_validates_before_atomic_activation_and_keeps_old_release(cml_paths, monkeypatch):
    project, environment, environment_root = cml_paths
    old_marker = environment / "bin/old-marker"
    old_marker.parent.mkdir(parents=True)
    old_marker.write_text("previous", encoding="utf-8")
    calls = _successful_subprocess(monkeypatch, project / "site-packages")

    setup_cml._install_environment()

    assert environment.is_symlink()
    assert (environment / "bin/python").read_text(encoding="utf-8") == "python"
    assert any(path.name.startswith("env-legacy-") for path in environment_root.iterdir())
    assert any(
        path.read_text(encoding="utf-8") == "previous"
        for path in environment_root.rglob("old-marker")
    )
    assert not any(path.name.startswith(".staging-") for path in environment_root.iterdir())
    assert "--clear" not in calls[0]
    assert ".staging-" in calls[0][-1]
    assert ".staging-" in calls[1][0]


def test_failed_install_preserves_prior_active_environment_and_cleans_stage(cml_paths, monkeypatch):
    _, environment, environment_root = cml_paths
    old_marker = environment / "bin/old-marker"
    old_marker.parent.mkdir(parents=True)
    old_marker.write_text("previous", encoding="utf-8")

    def run(command, **kwargs):
        del kwargs
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("python", encoding="utf-8")
            return SimpleNamespace(stdout="")
        if command[1:3] == ["-m", "pip"]:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(setup_cml.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        setup_cml._install_environment()

    assert environment.is_dir()
    assert not environment.is_symlink()
    assert old_marker.read_text(encoding="utf-8") == "previous"
    assert list(environment_root.iterdir()) == []


def test_activation_failure_rolls_back_legacy_directory(cml_paths, monkeypatch):
    project, environment, environment_root = cml_paths
    old_marker = environment / "bin/old-marker"
    old_marker.parent.mkdir(parents=True)
    old_marker.write_text("previous", encoding="utf-8")
    _successful_subprocess(monkeypatch, project / "site-packages")
    real_replace = setup_cml.os.replace
    pointer_attempts = 0

    def fail_pointer_activation(source, destination):
        nonlocal pointer_attempts
        if Path(destination) == environment:
            pointer_attempts += 1
            if pointer_attempts == 1:
                raise OSError("simulated pointer failure")
        real_replace(source, destination)

    monkeypatch.setattr(setup_cml.os, "replace", fail_pointer_activation)

    with pytest.raises(OSError, match="simulated pointer failure"):
        setup_cml._install_environment()

    assert environment.is_dir()
    assert not environment.is_symlink()
    assert old_marker.read_text(encoding="utf-8") == "previous"
    assert not any(path.name.startswith(".active-") for path in environment_root.iterdir())
    assert not any(path.name.startswith("env-") for path in environment_root.iterdir())
