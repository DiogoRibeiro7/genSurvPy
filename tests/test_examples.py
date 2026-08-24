"""Run the example scripts and the Binder notebooks.

They are shipped in the repository and linked from the documentation, so they
are a promise like any other. Nothing ran them, and they rotted: two scripts
still passed a ``qmat`` argument removed long ago, one of them describing the
Gaussian-emission hidden Markov model that was never implemented, and a
notebook passed three coefficients to ``gen_tdcm`` where two have been correct
since 1.3.0.

Each runs in a temporary directory, because several write plots.

Marked slow: the plotting examples fit Kaplan-Meier curves.
"""

from __future__ import annotations

import json
import pathlib
import runpy
import warnings

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = sorted((REPO / "examples").glob("run_*.py"))
NOTEBOOKS = sorted((REPO / "examples" / "notebooks").glob("*.ipynb"))


@pytest.fixture(autouse=True)
def _headless_and_isolated(tmp_path, monkeypatch):
    """Plot to a file-backed backend, in a directory nothing else writes to."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(tmp_path)

    import matplotlib

    matplotlib.use("Agg")
    yield

    import matplotlib.pyplot as plt

    plt.close("all")


def _assert_no_deprecations(caught: list[warnings.WarningMessage], label: str) -> None:
    """A deprecation here means the example is teaching an outdated call."""
    deprecated = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "gen_surv" in str(w.filename)
    ]
    assert not deprecated, f"{label} uses a deprecated API: {deprecated}"


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_example_script_runs(script: pathlib.Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runpy.run_path(str(script), run_name="__main__")

    _assert_no_deprecations(caught, script.name)


@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(notebook: pathlib.Path) -> None:
    """Execute a notebook's code cells in order, as Binder would."""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            try:
                exec(compile(source, f"{notebook.name}#cell{index}", "exec"), namespace)
            except Exception as exc:  # noqa: BLE001 - the failure is the message
                pytest.fail(
                    f"{notebook.name} cell {index} raised {type(exc).__name__}: {exc}"
                )

    _assert_no_deprecations(caught, notebook.name)


def test_the_examples_are_actually_there() -> None:
    """Guard against the globs silently matching nothing."""
    assert len(SCRIPTS) >= 7, f"found only {len(SCRIPTS)} example scripts"
    assert len(NOTEBOOKS) == 3, f"found {len(NOTEBOOKS)} notebooks"


def test_notebooks_seed_their_generators() -> None:
    """A notebook without a seed cannot be reproduced by the reader.

    All three used ``np.random.seed(0)``, which does nothing here: no generator
    reads NumPy's global state.
    """
    for notebook in NOTEBOOKS:
        source = notebook.read_text(encoding="utf-8")
        assert "seed=" in source, f"{notebook.name} does not seed its generator"
        assert "np.random.seed" not in source, (
            f"{notebook.name} calls np.random.seed, which has no effect on any "
            "generator in this package"
        )
