"""``papers/`` holds third-party literature PDFs and must never be committed.

The research studies under ``research/`` are written against copyrighted papers
kept locally in ``papers/``. Our own notes, summaries and bibliographic
metadata belong in version control; the source PDFs do not.

A ``.gitignore`` entry alone is not enough. ``git add -f`` overrides it, a
rename can move a file out from under it, and neither leaves a signal that
anyone would notice at review time. These tests fail the build instead.

They are deliberately cheap and have no third-party dependencies, so they run
everywhere the rest of the suite does.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Anything that would carry a paper's contents into the repository under
#: another name. Checked against tracked paths, not the working tree.
PDF_SUFFIXES = {".pdf", ".djvu", ".epub"}


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and result.stderr.strip():
        pytest.skip(f"git unavailable or not a repository: {result.stderr.strip()}")
    return result.stdout


@pytest.fixture(scope="module")
def tracked_files() -> list[str]:
    listing = _git("ls-files")
    if not listing.strip():
        pytest.skip("no tracked files reported; not a git checkout")
    return [line.strip() for line in listing.splitlines() if line.strip()]


def test_gitignore_still_excludes_papers() -> None:
    """The entry itself, so deleting it is a failing test rather than a silent regression."""
    gitignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    entries = {line.strip() for line in gitignore.splitlines()}

    assert "papers/" in entries, (
        "'papers/' is no longer in .gitignore. It holds third-party "
        "literature PDFs that must not reach GitHub."
    )


def test_git_reports_papers_as_ignored() -> None:
    """What .gitignore says and what git does are not always the same thing.

    A later negation pattern, or a `.gitignore` deeper in the tree, could
    re-include the directory. Ask git directly.
    """
    output = _git("check-ignore", "-v", "papers/")
    assert "papers/" in output, (
        "git does not report papers/ as ignored. Some other pattern may be "
        "re-including it; run `git check-ignore -v papers/` to see which."
    )


def test_no_file_below_papers_is_tracked(tracked_files: list[str]) -> None:
    """The rule that actually matters, and the one `git add -f` can break."""
    offenders = [path for path in tracked_files if path.startswith("papers/")]

    assert not offenders, (
        "These files under papers/ are tracked and must be removed from the "
        "index before this can be pushed:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse `git rm --cached <path>` to untrack without deleting."
    )


def test_no_pdfs_are_tracked_anywhere(tracked_files: list[str]) -> None:
    """Copying a PDF out of papers/ into a tracked directory evades the rule above.

    The package and its research directories have no legitimate reason to
    commit a PDF. If one is ever needed -- a figure exported for the
    manuscript, say -- prefer a vector source that is not a paper, or add an
    explicit exception here with the reason.
    """
    offenders = [
        path
        for path in tracked_files
        if pathlib.PurePath(path).suffix.lower() in PDF_SUFFIXES
    ]

    assert not offenders, (
        "Document files are tracked. If any of these came out of papers/, "
        "they are copyrighted and must not be committed:\n  " + "\n  ".join(offenders)
    )
