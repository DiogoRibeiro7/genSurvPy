#!/usr/bin/env python3
"""Check that pyproject version matches the latest git tag."""
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def pyproject_version() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data["tool"]["poetry"]["version"]


def latest_tag() -> str | None:
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], cwd=ROOT, text=True
        ).strip()
        return tag.lstrip("v")
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    tag = latest_tag()
    version = pyproject_version()

    if not tag:
        print("No git tag found", file=sys.stderr)
        return 1

    if version != tag:
        print(
            f"Version mismatch: pyproject.toml has {version} but latest tag is {tag}",
            file=sys.stderr,
        )
        return 1

    print(f"Version matches latest tag: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
