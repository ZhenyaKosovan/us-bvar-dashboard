"""Install the locked application environment into the CML project volume."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".cml-venv"
ENVIRONMENT_ROOT = ROOT / ".cml-venvs"
REQUIREMENTS = ROOT / "requirements-cml.txt"


def _path_exists(path: Path) -> bool:
    """Return true for normal paths and broken symlinks."""

    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _versioned_path(prefix: str) -> Path:
    return ENVIRONMENT_ROOT / f"{prefix}{uuid.uuid4().hex}"


def _activate(release: Path) -> None:
    """Atomically point the launcher at *release*, preserving a legacy directory."""

    target = os.path.relpath(release, ENVIRONMENT.parent)
    pointer = _versioned_path(".active-")
    os.symlink(target, pointer)
    previous_directory: Path | None = None

    try:
        if ENVIRONMENT.is_symlink():
            # Replacing a symlink is atomic and leaves its target available for rollback.
            os.replace(pointer, ENVIRONMENT)
            return

        if _path_exists(ENVIRONMENT):
            # Older setup versions created a real .cml-venv directory. Move it into the
            # versioned store before installing the new pointer so it is never deleted.
            previous_directory = _versioned_path("env-legacy-")
            os.replace(ENVIRONMENT, previous_directory)

        try:
            os.replace(pointer, ENVIRONMENT)
        except OSError:
            if _path_exists(ENVIRONMENT):
                _remove_path(ENVIRONMENT)
            if previous_directory is not None:
                os.replace(previous_directory, ENVIRONMENT)
            raise
    except OSError:
        if _path_exists(pointer):
            _remove_path(pointer)
        raise


def _install_environment() -> None:
    """Build, validate, and activate a new environment without touching the active one."""

    ENVIRONMENT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=ENVIRONMENT_ROOT))
    release: Path | None = None
    activated = False

    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(stage)],
            cwd=ROOT,
            check=True,
        )
        environment_python = stage / "bin/python"
        subprocess.run(
            [
                str(environment_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "--requirement",
                str(REQUIREMENTS),
            ],
            cwd=ROOT,
            check=True,
        )
        site_packages = subprocess.run(
            [
                str(environment_python),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not site_packages:
            raise RuntimeError("The staged CML environment did not report site-packages.")
        Path(site_packages, "us_bvar_dashboard.pth").write_text(
            f"{ROOT / 'src'}\n", encoding="utf-8"
        )
        subprocess.run(
            [str(environment_python), "-c", "import shiny, scipy, pandas, us_bvar"],
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
        )

        release = _versioned_path("env-")
        os.replace(stage, release)
        stage = None
        _activate(release)
        activated = True
    finally:
        # A failed build or activation must not remove or replace the prior active target.
        if not activated and release is not None and _path_exists(release):
            _remove_path(release)
        # This also removes a partially-created venv when venv/pip/validation fails.
        if stage is not None and _path_exists(stage):
            _remove_path(stage)


def main() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        raise SystemExit("Use a CML Python 3.11, 3.12, or 3.13 Runtime.")
    if not REQUIREMENTS.is_file():
        raise SystemExit("requirements-cml.txt is missing from the project.")

    _install_environment()
    print(f"CML environment is ready at {ENVIRONMENT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
