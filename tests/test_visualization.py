import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("lifelines")
pytest.importorskip("gen_surv")

import pandas as pd
import typer

from gen_surv import generate
from gen_surv.cli import visualize
from gen_surv.visualization import (
    describe_survival,
    plot_covariate_effect,
    plot_hazard_comparison,
    plot_survival_curve,
)


def test_plot_survival_curve_runs():
    df = generate(
        model="cphm",
        n=10,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=2.0,
    )
    fig, ax = plot_survival_curve(df)
    assert fig is not None
    assert ax is not None


def test_plot_survival_curve_by_group_runs():
    df = generate(
        model="cphm",
        n=10,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=2.0,
    )
    df["group"] = (df["X0"] > df["X0"].median()).astype(int)
    fig, ax = plot_survival_curve(df, group_col="group")
    assert fig is not None
    assert ax is not None


def test_plot_hazard_comparison_runs():
    df1 = generate(
        model="cphm",
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
    )
    df2 = generate(
        model="aft_weibull",
        n=5,
        beta=[0.5],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=1.0,
    )
    models = {"cphm": df1, "aft_weibull": df2}
    fig, ax = plot_hazard_comparison(models)
    assert fig is not None
    assert ax is not None


def test_plot_covariate_effect_runs():
    df = generate(
        model="cphm",
        n=10,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=2.0,
    )
    fig, ax = plot_covariate_effect(df, covariate_col="X0", n_groups=2)
    assert fig is not None
    assert ax is not None


def test_describe_survival_summary():
    df = generate(
        model="cphm",
        n=10,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=2.0,
    )
    summary = describe_survival(df)
    expected_metrics = [
        "Total Observations",
        "Number of Events",
        "Number Censored",
        "Event Rate",
        "Median Survival Time",
        "Min Time",
        "Max Time",
        "Mean Time",
    ]
    assert list(summary["Metric"]) == expected_metrics
    assert summary.shape[0] == len(expected_metrics)


def test_cli_visualize(tmp_path, capsys):
    df = pd.DataFrame({"time": [1, 2, 3], "status": [1, 0, 1]})
    csv_path = tmp_path / "d.csv"
    df.to_csv(csv_path, index=False)
    out_file = tmp_path / "out.png"
    visualize(
        str(csv_path),
        time_col="time",
        status_col="status",
        group_col=None,
        output=str(out_file),
    )
    assert out_file.exists()
    captured = capsys.readouterr()
    assert "Plot saved to" in captured.out


def test_cli_visualize_missing_column(tmp_path, capsys):
    df = pd.DataFrame({"time": [1, 2], "event": [1, 0]})
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    with pytest.raises(typer.Exit):
        visualize(
            str(csv_path),
            time_col="time",
            status_col="status",
            group_col=None,
        )
    captured = capsys.readouterr()
    assert "Status column 'status' not found in data" in captured.out


def test_cli_visualize_missing_time(tmp_path, capsys):
    df = pd.DataFrame({"t": [1, 2], "status": [1, 0]})
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    with pytest.raises(typer.Exit):
        visualize(str(path), time_col="time", status_col="status")
    captured = capsys.readouterr()
    assert "Time column 'time' not found in data" in captured.out


def test_cli_visualize_missing_group(tmp_path, capsys):
    df = pd.DataFrame({"time": [1], "status": [1], "x": [0]})
    path = tmp_path / "d2.csv"
    df.to_csv(path, index=False)
    with pytest.raises(typer.Exit):
        visualize(str(path), time_col="time", status_col="status", group_col="group")
    captured = capsys.readouterr()
    assert "Group column 'group' not found in data" in captured.out


def test_cli_visualize_import_error(monkeypatch, tmp_path, capsys):
    """visualize exits when matplotlib is missing."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):  # simulate missing dependency
            raise ImportError("no matplot")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    csv_path = tmp_path / "d.csv"
    pd.DataFrame({"time": [1], "status": [1]}).to_csv(csv_path, index=False)
    with pytest.raises(typer.Exit):
        visualize(str(csv_path))
    captured = capsys.readouterr()
    assert "Visualization requires matplotlib" in captured.out


def test_cli_visualize_read_error(monkeypatch, tmp_path, capsys):
    """visualize handles CSV read failures gracefully."""
    monkeypatch.setattr(
        "pandas.read_csv", lambda *a, **k: (_ for _ in ()).throw(Exception("boom"))
    )
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("time,status\n1,1\n")
    with pytest.raises(typer.Exit):
        visualize(str(csv_path))
    captured = capsys.readouterr()
    assert "Error loading CSV file" in captured.out
