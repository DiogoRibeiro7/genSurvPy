"""Report and gate scripts retain the original artifact failure."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
from dataexcept import DataLoadingError, FileWriteError
from survival_misspec.artifact_io import read_bytes, read_json, write_text


def test_missing_gate_artifact_preserves_source(tmp_path) -> None:
    path = tmp_path / "grid_audit_cells.json"

    with pytest.raises(DataLoadingError, match="grid_audit_cells.json") as error:
        read_json(path)

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_malformed_gate_artifact_preserves_decoder_error(tmp_path) -> None:
    path = tmp_path / "grid_audit_cells.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(DataLoadingError, match="grid_audit_cells.json") as error:
        read_json(path)

    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_unreadable_gate_digest_preserves_source(tmp_path) -> None:
    path = tmp_path / "missing.parquet"

    with pytest.raises(DataLoadingError, match="missing.parquet") as error:
        read_bytes(path)

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_report_write_preserves_destination(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("regular file", encoding="utf-8")
    target = blocked / "hypotheses.tex"

    with pytest.raises(FileWriteError, match="hypotheses.tex") as error:
        write_text(target, "\\newcommand{\\HOne}{yes}")

    assert isinstance(error.value.__cause__, OSError)


def test_figure_writer_closes_plot_after_filesystem_failure(tmp_path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    save_figure = runpy.run_path(str(script))["_save"]

    import matplotlib.pyplot as plt

    figure = plt.figure()
    blocked = tmp_path / "blocked"
    blocked.write_text("regular file", encoding="utf-8")

    with pytest.raises(FileWriteError, match="blocked") as error:
        save_figure(figure, "figure1", blocked, exploratory=False)

    assert isinstance(error.value.__cause__, OSError)
    assert not plt.fignum_exists(figure.number)
