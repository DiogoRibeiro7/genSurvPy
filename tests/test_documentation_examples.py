"""Execute every Python example in the documentation and the README.

Documentation drifts silently: an argument is renamed, a column moves, and the
example that used to work raises for the first reader who tries it. Worse, a
pasted output stops being what the code prints and nobody notices, because
nothing runs it. These tests run the examples and compare their output, so
drift fails a build instead.

Blocks on a page share a namespace, which is how a reader follows them: an
example that depends on a variable defined three sections earlier still works,
and one that depends on a variable that no longer exists does not.

A ``text`` block directly beneath a printing example is treated as that
example's output and compared. Separated by prose, it belongs to a different
paragraph and is left alone.

Some blocks are documentation rather than code -- call signatures, errors shown
deliberately, and two model fits too slow to belong in a test run. They are
listed in ``NOT_EXECUTED`` with a reason, and ``test_no_stale_exclusions``
fails if an entry stops matching anything, so nothing sits there being skipped
forever.

Marked slow: it imports lifelines and fits models.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import re
import textwrap

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = pathlib.Path(__file__).resolve().parent.parent

# Fences, including those indented inside tabs and admonitions.
FENCE = re.compile(r"^[ \t]*```(python|text)\n(.*?)^[ \t]*```", re.S | re.M)

#: Blocks that are not executed, keyed by page, identified by a distinctive
#: fragment of the block. Every entry carries its reason.
NOT_EXECUTED: dict[str, list[str]] = {
    # Call signatures, shown so a reader can see the argument order.
    "docs/models/cphm.md": ["gen_cphm(n, model_cens"],
    "docs/models/aft.md": ["gen_aft_log_normal(n, beta"],
    "docs/models/cmm.md": ["gen_cmm(n, model_cens"],
    "docs/models/thmm.md": ["gen_thmm(n, model_cens"],
    "docs/models/competing-risks.md": ["gen_competing_risks(n, n_risks"],
    "docs/models/mixture-cure.md": ["gen_mixture_cure(n, cure_fraction"],
    "docs/models/piecewise-exponential.md": [
        "gen_piecewise_exponential(n, breakpoints"
    ],
    "docs/models/recurrent-events.md": ["gen_recurrent_events(n, process"],
    "docs/guides/covariates.md": [
        'generate(model="...", covariate_dist',
        # An error shown on purpose: the message is the documented output.
        'covariate_params={"mean": 0.0}',
    ],
    # Errors shown on purpose.
    "docs/models/index.md": ['generate(model="weibull")'],
    "docs/guides/baselines.md": ["WeibullBaseline(shape=0.0"],
    # Uses `...` deliberately, to keep the point about generators short.
    "docs/getting-started/reproducibility.md": ["first  = generate(..., seed=rng)"],
    "docs/models/tdcm.md": [
        "gen_tdcm(n, dist, corr",
        # A time-varying Cox fit over 30,000 subjects: three minutes on its own,
        # which is most of this module's runtime. Its output was verified by
        # hand and is quoted on the page; the same fit at a workable size is
        # exercised by tests/test_tdcm_crossover.py.
        "CoxTimeVaryingFitter().fit(pd.concat",
    ],
}


def _pages() -> list[pathlib.Path]:
    return sorted(REPO.glob("docs/**/*.md")) + [REPO / "README.md"]


def _blocks(page: pathlib.Path) -> list[tuple[str, str, int, int]]:
    """Return ``(language, body, start, end)`` for each fence on the page."""
    source = page.read_text(encoding="utf-8")
    return [
        (m.group(1), textwrap.dedent(m.group(2)), m.start(), m.end())
        for m in FENCE.finditer(source)
    ]


def _directly_follows(code_end: int, text_start: int, source: str) -> bool:
    """Whether a text fence is the code fence's output rather than a later aside."""
    return source[code_end:text_start].strip() == ""


def _excluded(rel: str, body: str) -> bool:
    return any(marker in body for marker in NOT_EXECUTED.get(rel, []))


def _normalise(text: str) -> str:
    return "\n".join(" ".join(line.split()) for line in text.strip().splitlines())


@pytest.mark.slow
@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.name)
def test_examples_run_and_their_output_is_what_is_documented(
    page: pathlib.Path, tmp_path, monkeypatch
) -> None:
    """Run each example in order, then check any output pasted beneath it."""
    monkeypatch.chdir(tmp_path)  # examples that write files stay in a temp dir

    rel = page.relative_to(REPO).as_posix()
    source = page.read_text(encoding="utf-8")
    blocks = _blocks(page)

    # ``dataclass`` resolves ``__module__`` through ``sys.modules``, so the
    # namespace needs a name that is actually there.
    namespace: dict[str, object] = {"__name__": "__main__"}

    for index, (language, body, _, end) in enumerate(blocks):
        if language != "python" or _excluded(rel, body):
            continue

        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(compile(body, f"{rel}#block{index}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - the failure is the message
            pytest.fail(
                f"{rel}#block{index} raised {type(exc).__name__}: {exc}\n"
                f"  the block starts: {body.strip().splitlines()[0]}"
            )

        printed = captured.getvalue().strip()
        follows = blocks[index + 1] if index + 1 < len(blocks) else None
        if (
            not printed
            or follows is None
            or follows[0] != "text"
            or not _directly_follows(end, follows[2], source)
        ):
            continue

        assert _normalise(printed) == _normalise(follows[1]), (
            f"{rel}#block{index}: the documented output no longer matches what "
            f"the example prints.\n--- documented ---\n{follows[1].strip()}\n"
            f"--- actual ---\n{printed}"
        )


def test_no_stale_exclusions() -> None:
    """Every entry in ``NOT_EXECUTED`` must still match a block.

    Without this, an example fixed long ago would sit in the list being skipped,
    and nobody would notice it had become executable.
    """
    stale: list[str] = []
    for rel, markers in NOT_EXECUTED.items():
        page = REPO / rel
        assert page.exists(), f"{rel} is listed but does not exist"
        bodies = [
            body for language, body, _, _ in _blocks(page) if language == "python"
        ]
        stale.extend(
            f"{rel}: {marker!r}"
            for marker in markers
            if not any(marker in body for body in bodies)
        )

    assert not stale, "stale entries in NOT_EXECUTED: " + ", ".join(stale)


def test_every_page_is_covered() -> None:
    """The parametrisation must actually reach the documentation."""
    pages = _pages()

    assert len(pages) > 30
    assert (REPO / "README.md") in pages
    assert any(p.name == "quickstart.md" for p in pages)
