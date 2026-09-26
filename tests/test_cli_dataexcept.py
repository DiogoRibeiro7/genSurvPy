"""DataExcept context at the CLI's external data and file boundaries."""

import pandas as pd
import pytest
import typer
from dataexcept import DataLoadingError, FileWriteError, SchemaMismatchError

from gen_surv.cli import _load_csv, dataset, visualize


def test_csv_read_error_retains_source_and_cause(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(DataLoadingError) as captured:
        _load_csv(str(path))

    assert captured.value.source == str(path)
    assert isinstance(captured.value.__cause__, FileNotFoundError)


def test_dataset_write_error_exits_with_typed_cause(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "gen_surv.cli.generate",
        lambda **kwargs: pd.DataFrame({"time": [1], "status": [1]}),
    )
    path = tmp_path / "missing-directory" / "data.csv"

    with pytest.raises(typer.Exit) as captured:
        dataset(model="cphm", n=1, output=str(path))

    error = captured.value.__cause__
    assert isinstance(error, FileWriteError)
    assert error.path == str(path)
    assert isinstance(error.__cause__, OSError)
    assert "Error writing CSV file" in capsys.readouterr().out


def test_visualize_missing_column_retains_schema(tmp_path, capsys):
    pytest.importorskip("lifelines")
    path = tmp_path / "data.csv"
    pd.DataFrame({"time": [1, 2], "event": [1, 0]}).to_csv(path, index=False)

    with pytest.raises(typer.Exit) as captured:
        visualize(str(path), time_col="time", status_col="status", group_col=None)

    error = captured.value.__cause__
    assert isinstance(error, SchemaMismatchError)
    assert error.expected == "Status column 'status'"
    assert "'event'" in error.found
    assert "Status column 'status' not found" in capsys.readouterr().out


def test_visualize_plot_write_failure_closes_figure(tmp_path, monkeypatch, capsys):
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("lifelines")
    path = tmp_path / "data.csv"
    pd.DataFrame({"time": [1, 2], "status": [1, 0]}).to_csv(path, index=False)
    figure, axes = plt.subplots()
    monkeypatch.setattr(
        "gen_surv.visualization.plot_survival_curve", lambda **kwargs: (figure, axes)
    )
    failure = PermissionError("output denied")

    def fail_savefig(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(plt, "savefig", fail_savefig)
    output = str(tmp_path / "plot.png")

    with pytest.raises(typer.Exit) as captured:
        visualize(
            str(path),
            time_col="time",
            status_col="status",
            group_col=None,
            output=output,
        )

    error = captured.value.__cause__
    assert isinstance(error, FileWriteError)
    assert error.path == output
    assert error.__cause__ is failure
    assert not plt.fignum_exists(figure.number)
    assert "Error writing plot" in capsys.readouterr().out
