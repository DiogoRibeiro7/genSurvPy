#!/usr/bin/env python
"""pyproject_updater.py

Update dependency constraints in `pyproject.toml` to the **latest** versions from PyPI.

Features
--------
- Reads both Poetry ([tool.poetry]) and PEP 621 ([project]) layouts.
- Preserves formatting and comments using tomlkit.
- Fetches the latest version from PyPI (skips yanked releases; pre-releases optional).
- Multiple update strategies: exact | caret | tilde | floor.
- By default respects the current major version unless --allow-major is set.
- Can target specific groups (Poetry group or PEP 621 optional-dependencies) and/or specific packages.
- Dry-run `--check` prints a unified diff without writing.

Usage
-----
python scripts/pyproject_updater.py upgrade \
    --strategy caret \
    --groups main,dev \
    --only numpy,pandas \
    --respect-major \
    --no-prerelease \
    --check

Requirements
------------
pip install tomlkit packaging
(Uses only stdlib for HTTP via urllib.request.)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import tomlkit
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version


@dataclass(frozen=True)
class Options:
    strategy: str  # one of: exact, caret, tilde, floor
    allow_major: bool  # if False, keep within current major
    include_prerelease: bool  # if True, accept pre-releases
    groups: list[str]  # e.g., ["main", "dev"]
    only: list[str] | None  # package name filters (normalized)
    check: bool  # dry-run
    file: Path  # pyproject.toml path
    timeout: float  # HTTP timeout


# ---------- TOML helpers ----------


def _read_doc(path: Path):
    text = path.read_text(encoding="utf-8")
    return text, tomlkit.parse(text)


def _write_or_diff(path: Path, before: str, after: str, check: bool) -> int:
    if check:
        diff = "".join(
            unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=str(path),
                tofile=str(path),
            )
        )
        sys.stdout.write(diff)
        return 0
    path.write_text(after, encoding="utf-8")
    return 0


def _layout(doc) -> str:
    # Prefer Poetry if both exist
    if "tool" in doc and isinstance(doc["tool"], dict) and "poetry" in doc["tool"]:
        return "poetry"
    if "project" in doc:
        return "pep621"
    raise ValueError("Unsupported pyproject: neither [tool.poetry] nor [project] found.")


# ---------- PyPI version lookup ----------


def _normalize_pkg_name(name: str) -> str:
    """Normalize per PEP 503 for PyPI URLs: lowercase and replace `_`/`.` with `-`.
    Keep extras separate (e.g., 'foo[bar]') — we strip extras for lookup.
    """
    base = name.split("[", 1)[0]
    return base.lower().replace("_", "-").replace(".", "-")


def _fetch_pypi_versions(name: str, timeout: float) -> dict[str, bool]:
    """Return ``{version_str: is_yanked}`` for package *name* from PyPI."""
    url = f"https://pypi.org/pypi/{_normalize_pkg_name(name)}/json"
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}

    versions: dict[str, bool] = {}
    releases = data.get("releases", {}) or {}
    for ver_str, files in releases.items():
        # files is a list of distributions; consider version yanked if all files are yanked
        if not isinstance(files, list) or len(files) == 0:
            continue
        all_yanked = all(bool(f.get("yanked", False)) for f in files if isinstance(f, dict))
        versions[ver_str] = not all_yanked
    return versions


def _select_latest_version(versions: dict[str, bool], include_prerelease: bool) -> Version | None:
    """Pick the highest non-yanked Version. If include_prerelease=False, prefer finals."""
    valid: list[Version] = []
    for ver_str, not_yanked in versions.items():
        if not not_yanked:
            continue
        try:
            v = Version(ver_str)
        except InvalidVersion:
            continue
        if (not include_prerelease) and v.is_prerelease:
            continue
        valid.append(v)
    if not valid:
        return None
    return max(valid)


# ---------- Constraint mapping ----------


def _poetry_string_for_strategy(v: Version, strategy: str) -> str:
    if strategy == "exact":
        return f"=={v}"
    if strategy == "caret":
        return f"^{v}"
    if strategy == "tilde":
        return f"~{v}"
    if strategy == "floor":
        return f">={v}"
    raise ValueError(f"Unknown strategy: {strategy}")


def _pep440_string_for_strategy(v: Version, strategy: str) -> str:
    if strategy == "exact":
        return f"=={v}"
    if strategy == "floor":
        return f">={v}"
    if strategy == "caret":
        # compatible with current MAJOR; upper bound next major
        upper = f"{v.major + 1}.0.0"
        return f">={v},<{upper}"
    if strategy == "tilde":
        # compatible with current MINOR; upper bound next minor
        upper = f"{v.major}.{v.minor + 1}.0"
        return f">={v},<{upper}"
    raise ValueError(f"Unknown strategy: {strategy}")


def _respect_major_allowed(current_spec: str | None, latest: Version, allow_major: bool) -> bool:
    """If allow_major is False and current_spec indicates a major cap, avoid bumping across majors.
    Heuristic: extract existing max major from spec if present; otherwise compare against any pinned/ranged major.
    """
    if allow_major:
        return True
    if not current_spec:
        return latest.major == latest.major  # trivial True
    # Try to parse the requirement to see any existing max
    try:
        req = (
            Requirement(f"pkg {current_spec}")
            if " " in current_spec
            else Requirement(f"pkg {current_spec}")
        )
    except Exception:
        # Fallback: if spec starts with ^ or ~ (Poetry), infer major from latest string of spec if present
        if current_spec.startswith("^") or current_spec.startswith("~"):
            # Keep within the current major implied by the spec's base number
            try:
                base = Version(current_spec.lstrip("^~=>=<=!~^ "))
                return latest.major <= base.major
            except Exception:
                return True
        return True

    # If there is an upper bound like <2.0.0, prevent crossing it.
    for sp in req.specifier:
        if sp.operator in ("<", "<="):
            try:
                upper = Version(sp.version)
                if upper.major <= latest.major:
                    return False
            except InvalidVersion:
                pass
    return True


# ---------- Dependency iteration & rewriting ----------


@dataclass
class DepRef:
    layout: str  # "poetry" or "pep621"
    group: str  # "main" or group name
    name: str  # raw name as in file (may include extras)
    current_spec: str | None  # string spec; None if path/git or table
    location: tuple  # references for writing (table/array and key/index)


def _iter_poetry_deps(doc, groups: Iterable[str]) -> Iterable[DepRef]:
    tool = doc.setdefault("tool", tomlkit.table())
    poetry = tool.setdefault("poetry", tomlkit.table())

    def emit_from_table(tbl, group: str):
        if not isinstance(tbl, dict):
            return
        for k, v in list(tbl.items()):
            # Skip python pseudo-dep
            if k == "python":
                continue
            if isinstance(v, str):
                yield DepRef("poetry", group, k, v, (tbl, k))
            elif isinstance(v, dict):
                # Support {version="..."}; skip non-PyPI (path, git)
                ver = v.get("version")
                if isinstance(ver, str):
                    yield DepRef("poetry", group, k, ver, (v, "version"))
                else:
                    continue  # non-versioned (git/path) -> skip
            else:
                continue

    wanted = set(groups)
    if "main" in wanted or not wanted:
        deps = poetry.get("dependencies", {})
        yield from emit_from_table(deps, "main")

    # dev-dependencies (legacy)
    if "dev" in wanted:
        dev = poetry.get("dev-dependencies", {})
        yield from emit_from_table(dev, "dev")

    # named groups
    group_tbl = poetry.get("group", {})
    if isinstance(group_tbl, dict):
        for gname, gtbl in group_tbl.items():
            if wanted and gname not in wanted:
                continue
            if isinstance(gtbl, dict):
                deps = gtbl.get("dependencies", {})
                yield from emit_from_table(deps, gname)


def _iter_pep621_deps(doc, groups: Iterable[str]) -> Iterable[DepRef]:
    project = doc.setdefault("project", tomlkit.table())
    groups_set = set(groups)

    def emit_from_array(arr, group: str):
        if not isinstance(arr, tomlkit.items.Array):
            return
        for idx, item in enumerate(list(arr)):
            if not isinstance(item, str):
                continue
            try:
                req = Requirement(item)
            except Exception:
                continue
            spec = str(req.specifier) if req.specifier else None
            # Keep original text shape; we’ll overwrite the whole string at index
            yield DepRef("pep621", group, req.name, spec, (arr, idx, req))

    # main deps
    if not groups_set or "main" in groups_set:
        arr = project.setdefault("dependencies", tomlkit.array())
        emit = list(emit_from_array(arr, "main"))
        yield from emit

    # optional groups
    opt = project.setdefault("optional-dependencies", tomlkit.table())
    if isinstance(opt, dict):
        for gname, arr in opt.items():
            if groups_set and gname not in groups_set:
                continue
            emit = list(emit_from_array(arr, gname))
            yield from emit


def _set_dep_spec(dep: DepRef, new_spec: str):
    """Write back new spec to the TOML document at the stored location."""
    if dep.layout == "poetry":
        tbl, key = dep.location  # type: ignore[assignment]
        if isinstance(tbl, dict):
            # If it was a string spec: overwrite with string.
            if isinstance(tbl.get(key), str):
                tbl[key] = new_spec
            elif isinstance(tbl.get(key), dict):
                tbl[key]["version"] = new_spec
            else:
                tbl[key] = new_spec
    else:
        arr, idx, req = dep.location  # type: ignore[assignment]
        if isinstance(arr, tomlkit.items.Array):
            # Rebuild requirement string with new spec; keep extras/markers
            name = req.name
            extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
            markers = f"; {req.marker}" if req.marker else ""
            arr[idx] = f"{name}{extras} {new_spec}{markers}".strip()


# ---------- Main upgrade routine ----------


def upgrade(pyproject: Path, opts: Options) -> int:
    before_text, doc = _read_doc(pyproject)
    layout = _layout(doc)

    # Which groups to consider by default
    groups = opts.groups or ["main", "dev"]  # include Poetry dev by default
    only_norm = {_normalize_pkg_name(n) for n in (opts.only or [])}

    # Iterate deps
    iterator = _iter_poetry_deps if layout == "poetry" else _iter_pep621_deps
    changed = 0

    for dep in iterator(doc, groups):
        base_norm = _normalize_pkg_name(dep.name)
        if only_norm and base_norm not in only_norm:
            continue
        # Skip obviously non-PyPI
        if dep.current_spec is None:
            continue

        # Respect-major check (heuristic against crossing major caps)
        # We perform check after we fetch latest.
        versions = _fetch_pypi_versions(dep.name, opts.timeout)
        latest = _select_latest_version(versions, opts.include_prerelease)
        if latest is None:
            continue

        if not _respect_major_allowed(dep.current_spec, latest, opts.allow_major):
            # If not allowed, try to keep within current major by taking the max version < next major.
            target_major = None
            # Guess current allowed major from current spec or dep.current_spec base
            try:
                if dep.current_spec.startswith(("^", "~")):
                    target_major = Version(dep.current_spec.lstrip("^~ =><")).major
            except Exception:
                pass
            if target_major is not None:
                # pick highest < target_major+1.0.0
                candidates = [Version(v) for v, ok in versions.items() if ok]
                within = [
                    v
                    for v in candidates
                    if (
                        v.major == target_major and (opts.include_prerelease or not v.is_prerelease)
                    )
                ]
                if within:
                    latest = max(within)

        # Compute new spec string according to layout/strategy
        if layout == "poetry":
            new_spec = _poetry_string_for_strategy(latest, opts.strategy)
        else:
            new_spec = _pep440_string_for_strategy(latest, opts.strategy)

        # If spec already implies >= latest (exact match for exact), skip writing.
        if dep.current_spec and dep.current_spec.strip() == new_spec.strip():
            continue

        _set_dep_spec(dep, new_spec)
        changed += 1

    after_text = tomlkit.dumps(doc)
    if before_text == after_text:
        # Nothing to do; still honor --check by printing empty diff (no output).
        return 0
    return _write_or_diff(pyproject, before_text, after_text, opts.check)


def parse_args(argv: list[str] | None = None) -> Options:
    p = argparse.ArgumentParser(
        description="Upgrade pyproject dependency constraints to latest from PyPI."
    )
    p.add_argument("--file", default="pyproject.toml", help="Path to pyproject.toml")
    p.add_argument(
        "--strategy",
        choices=["exact", "caret", "tilde", "floor"],
        default="caret",
        help="How to express the updated constraint.",
    )
    p.add_argument(
        "--allow-major", action="store_true", help="Allow bumping to a new MAJOR version."
    )
    p.add_argument(
        "--respect-major",
        dest="allow_major",
        action="store_false",
        help="(default) Keep within the current major if possible.",
    )
    p.set_defaults(allow_major=False)
    p.add_argument(
        "--pre",
        "--include-prerelease",
        dest="include_prerelease",
        action="store_true",
        help="Allow pre-releases when picking the latest.",
    )
    p.add_argument(
        "--no-prerelease",
        dest="include_prerelease",
        action="store_false",
        help="(default) Exclude pre-releases.",
    )
    p.set_defaults(include_prerelease=False)
    p.add_argument(
        "--groups",
        default="main,dev",
        help="Comma-separated groups: e.g., main,dev or analytics,docs",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated package names to update (normalized). Empty=all.",
    )
    p.add_argument("--check", action="store_true", help="Dry-run: show unified diff, do not write.")
    p.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout (seconds).")

    args = p.parse_args(argv)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    only = [n.strip() for n in args.only.split(",") if n.strip()] or None

    return Options(
        strategy=args.strategy,
        allow_major=args.allow_major,
        include_prerelease=args.include_prerelease,
        groups=groups,
        only=only,
        check=args.check,
        file=Path(args.file),
        timeout=args.timeout,
    )


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    return upgrade(opts.file, opts)


if __name__ == "__main__":
    raise SystemExit(main())
