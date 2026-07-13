"""Install the locked application environment into the CML project volume."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".cml-venv"
REQUIREMENTS = ROOT / "requirements-cml.txt"


def main() -> None:
    if sys.version_info < (3, 11) or sys.version_info >= (3, 14):
        raise SystemExit("Use a CML Python 3.11, 3.12, or 3.13 Runtime.")
    if not REQUIREMENTS.is_file():
        raise SystemExit("requirements-cml.txt is missing from the project.")

    subprocess.run(
        [sys.executable, "-m", "venv", "--clear", str(ENVIRONMENT)],
        cwd=ROOT,
        check=True,
    )
    environment_python = ENVIRONMENT / "bin/python"
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
    Path(site_packages, "us_bvar_dashboard.pth").write_text(f"{ROOT / 'src'}\n", encoding="utf-8")
    subprocess.run(
        [str(environment_python), "-c", "import shiny, scipy, pandas, us_bvar"],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )
    print(f"CML environment is ready at {ENVIRONMENT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
