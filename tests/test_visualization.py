import pandas as pd
import pytest
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
