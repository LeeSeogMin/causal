#!/usr/bin/env python3
"""
Course environment bootstrap for this repo.

- Creates a local venv (default: .venv/)
- Installs base requirements from requirements.txt
- Optionally installs extra packages for advanced chapters
- Verifies imports with --check

This script is meant to be run by students on their own machines.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _print_header() -> None:
    print("=" * 80)
    print("setup_env.py: course environment bootstrap")
    print("=" * 80)
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")


def create_venv(venv_dir: Path) -> None:
    if venv_dir.exists():
        return
    print(f"[1/4] Creating venv at: {venv_dir}")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=False, upgrade=False)
    builder.create(str(venv_dir))


def install_packages(python_exe: Path, packages: list[str], *, upgrade_pip: bool) -> None:
    if upgrade_pip:
        print("[2/4] Upgrading pip...")
        _run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    if not packages:
        return
    print("[3/4] Installing packages...")
    _run([str(python_exe), "-m", "pip", "install", *packages])


def check_imports(python_exe: Path) -> None:
    print("[4/4] Verifying key imports...")
    code = r"""
import importlib
import sys

mods = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "statsmodels",
    "networkx",
    "econml",
    "xgboost",
]

failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"OK  {m}")
    except Exception as e:
        failed.append((m, repr(e)))
        print(f"FAIL {m}: {e}")

if failed:
    print("\nSome packages failed to import:")
    for m, err in failed:
        print(f"- {m}: {err}")
    raise SystemExit(1)

print("\nAll key imports look good.")
print("python:", sys.version.split()[0])
"""
    _run([str(python_exe), "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up Python environment for this course repo.")
    parser.add_argument("--venv-dir", default=".venv", help="Virtualenv directory (default: .venv)")
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Do not create/use a venv; install into current Python environment (not recommended).",
    )
    parser.add_argument(
        "--extras",
        action="append",
        default=[],
        help="Install optional packages: torch, tensorflow, all (can be repeated).",
    )
    parser.add_argument("--no-upgrade-pip", action="store_true", help="Skip pip upgrade step.")
    parser.add_argument("--check", action="store_true", help="Only verify imports (no installs).")
    args = parser.parse_args()

    _print_header()

    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required.")
        return 2

    venv_dir = (ROOT / args.venv_dir).resolve()
    use_venv = not args.skip_venv

    if use_venv:
        create_venv(venv_dir)
        python_exe = _venv_python(venv_dir)
        if not python_exe.exists():
            print(f"ERROR: venv python not found at: {python_exe}")
            return 2
    else:
        python_exe = Path(sys.executable)
        print("WARNING: --skip-venv selected; installing into current Python environment.")

    base_reqs = _read_requirements(ROOT / "requirements.txt")

    extras = [e.strip().lower() for e in args.extras if e.strip()]
    want_all = "all" in extras
    optional_reqs = _read_requirements(ROOT / "requirements-optional.txt")

    selected_optional: list[str] = []
    if want_all:
        selected_optional = optional_reqs
    else:
        # Keep this small and predictable; advanced packages are intentionally opt-in.
        for name in ("torch", "tensorflow"):
            if name in extras:
                selected_optional.append(name)

    if args.check:
        check_imports(python_exe)
        return 0

    pkgs = base_reqs + selected_optional
    print(f"\nTarget python: {python_exe}")
    print(f"Will install {len(pkgs)} packages ({len(base_reqs)} base, {len(selected_optional)} optional).")

    install_packages(python_exe, pkgs, upgrade_pip=not args.no_upgrade_pip)
    check_imports(python_exe)

    print("\nNext steps:")
    if use_venv:
        if os.name == "nt":
            print(r"- Activate: .\.venv\Scripts\Activate.ps1")
        else:
            print("- Activate: source .venv/bin/activate")
    print("- Run a practice script, e.g.: python practice/chapter02/code/2-1-potential-outcomes.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

