# Literature

Our own notes, synthesis and bibliographic metadata. **Source PDFs never go
here.**

The PDFs live in `<repository root>/papers/`, which is gitignored and must
never reach GitHub. `tests/test_papers_not_tracked.py` in the root test suite
enforces that: it checks the `.gitignore` entry, what git actually reports,
that nothing under `papers/` is tracked, and that no PDF is tracked anywhere —
which catches copying one out to evade the rule. Force-adding a PDF fails three
of its four tests.

## Contents

- [`positioning.md`](positioning.md) — what this paper claims, what it does not,
  and which prior work it must differentiate against. Revised after the author's
  reading of the recent literature.
- `../paper/references.bib` — the bibliography, transcribed as supplied. DOIs
  are unverified.

## Rules

**Nothing is cited before it is read.** The bibliography exists so the entries
are recorded; it is not evidence that the work has been reviewed. Claims about
what a paper says belong here only after someone has opened it.

**No extracted text from a copyrighted paper is committed**, here or anywhere
else in the repository. Summaries in our own words are fine; passages are not.
