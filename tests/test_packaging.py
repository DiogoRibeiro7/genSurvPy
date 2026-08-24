"""The package must advertise its type information to downstream users.

`gen_surv` is fully annotated and mypy runs over it on every commit, but none of
that reached anyone installing it. Without a `py.typed` marker, PEP 561 tells a
type checker to ignore an installed package's inline annotations, so a
downstream `mypy` reported

    error: Cannot find implementation or library stub for module named "gen_surv"

and silently accepted `gen_cphm(n="not an integer", ...)`. With the marker, the
same call is reported as `Argument "n" ... has incompatible type "str";
expected "int"`.
"""

from __future__ import annotations

import inspect
import pathlib

import gen_surv


def _package_root() -> pathlib.Path:
    assert gen_surv.__file__ is not None
    return pathlib.Path(gen_surv.__file__).parent


def test_py_typed_marker_ships_with_the_package() -> None:
    """Checked on the imported package, so an install without it fails too."""
    marker = _package_root() / "py.typed"

    assert marker.is_file(), (
        "gen_surv/py.typed is missing. Without it a type checker ignores every "
        "annotation in the package, whatever the source says."
    )


def test_the_public_api_is_actually_annotated() -> None:
    """A marker promising types that are not there would be worse than none."""
    unannotated: list[str] = []

    for name in gen_surv.__all__:
        obj = getattr(gen_surv, name, None)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue

        for parameter in signature.parameters.values():
            if parameter.name in ("self", "cls", "args", "kwargs"):
                continue
            if parameter.annotation is inspect.Parameter.empty:
                unannotated.append(f"{name}({parameter.name})")

    assert not unannotated, f"unannotated public parameters: {unannotated}"


def test_public_functions_declare_a_return_type() -> None:
    missing: list[str] = []

    for name in gen_surv.__all__:
        obj = getattr(gen_surv, name, None)
        if not inspect.isfunction(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            continue
        if signature.return_annotation is inspect.Signature.empty:
            missing.append(name)

    assert not missing, f"public functions without a return annotation: {missing}"
