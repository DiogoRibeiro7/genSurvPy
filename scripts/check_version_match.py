#!/usr/bin/env python3
"""Keep ``pyproject.toml`` in sync with the latest git tag.

When run with ``--fix`` the script will create a git tag from the version
declared in ``pyproject.toml``. Supplying ``--write`` updates the
``pyproject.toml`` version to match the latest tag. Using both flags ensures
that whichever side is ahead becomes the single source of truth.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):  # pragma: no cover - stdlib alias
    import tomllib as tomli
else:  # pragma: no cover - python <3.11 fallback
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


def write_version(version: str) -> None:
    """Update ``pyproject.toml`` with *version* and push the change."""
    pyproject_path = ROOT / "pyproject.toml"
    content = pyproject_path.read_text()
    updated = re.sub(
        r'^version = "[^"]+"', f'version = "{version}"', content, flags=re.MULTILINE
    )
    pyproject_path.write_text(updated)
    subprocess.run(["git", "add", str(pyproject_path)], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"chore: bump version to {version}"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "push"], cwd=ROOT, check=True)
    print(f"✅ pyproject.toml updated to {version} and pushed.")


def _split(v: str) -> tuple[int, ...]:
    return tuple(int(part) for part in v.split("."))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync git tags and pyproject version")
    parser.add_argument(
        "--fix", action="store_true", help="Tag repo from pyproject version"
    )
    parser.add_argument(
        "--write", action="store_true", help="Update pyproject version from latest tag"
    )
    args = parser.parse_args()

    version = pyproject_version()
    tag = latest_tag()

    if not tag:
        print("⚠️  No git tag found.", file=sys.stderr)
        if args.fix:
            create_tag(version)
            return 0
        return 1

    if version != tag:
        print(
            f"❌ Version mismatch: pyproject.toml has {version} but latest tag is {tag}",
            file=sys.stderr,
        )
        if args.fix and _split(version) > _split(tag):
            create_tag(version)
            return 0
        if args.write and _split(tag) > _split(version):
            write_version(tag)
            return 0
        return 1

    print(f"✔️  Version matches latest tag: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
