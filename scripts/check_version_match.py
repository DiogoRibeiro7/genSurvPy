#!/usr/bin/env python3
"""Check that pyproject version matches the latest git tag. Optionally fix it by tagging."""
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib as tomli
else:
    import tomli

ROOT = Path(__file__).resolve().parents[1]


def pyproject_version() -> str:
    pyproject_path = ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data: Any = tomli.load(f)
    return cast(str, data["tool"]["poetry"]["version"])


def latest_tag() -> str:
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"], cwd=ROOT, text=True
        ).strip()
        return tag.lstrip("v")
    except subprocess.CalledProcessError:
        return ""


def create_tag(version: str) -> None:
    print(f"Tagging repository with version: v{version}")
    subprocess.run(["git", "tag", f"v{version}"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", f"v{version}"], cwd=ROOT, check=True)
    print(f"✅ Git tag v{version} created and pushed.")


def main() -> int:
    fix = "--fix" in sys.argv
    version = pyproject_version()
    tag = latest_tag()

    if not tag:
        print("⚠️  No git tag found.", file=sys.stderr)
        if fix:
            create_tag(version)
            return 0
        else:
            return 1

    if version != tag:
        print(
            f"❌ Version mismatch: pyproject.toml has {version} but latest tag is {tag}",
            file=sys.stderr,
        )
        if fix:
            create_tag(version)
            return 0
        return 1

    print(f"✔️  Version matches latest tag: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
