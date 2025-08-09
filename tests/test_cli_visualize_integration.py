from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("lifelines")
pytest.importorskip("gen_surv")

from typer.testing import CliRunner

from gen_surv import generate
from gen_surv.cli import app


def test_visualize_cli_generates_plot(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    plot_path = tmp_path / "plot.png"
    df = generate(
        model="cphm",
        n=10,
        beta=0.5,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=0.7,
        seed=1234,
    )
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app, ["visualize", str(csv_path), "--output", str(plot_path)]
    )
    assert result.exit_code == 0
    assert plot_path.exists()


def test_visualize_cli_missing_status(tmp_path) -> None:
    """CLI exits with error when status column is absent."""
    csv_path = tmp_path / "data.csv"
    generate(
        model="cphm",
        n=3,
        beta=0.1,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=0.5,
    ).drop(columns=["status"]).to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(app, ["visualize", str(csv_path)])
    assert result.exit_code != 0
    assert "Status column" in result.stdout


def test_visualize_cli_missing_time(tmp_path) -> None:
    """CLI exits with error when time column is absent."""
    csv_path = tmp_path / "data.csv"
    generate(
        model="cphm",
        n=3,
        beta=0.1,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=0.5,
    ).drop(columns=["time"]).to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(app, ["visualize", str(csv_path)])
    assert result.exit_code != 0
    assert "Time column" in result.stdout


def test_visualize_cli_custom_columns(tmp_path) -> None:
    """CLI handles custom time/status column names."""
    csv_path = tmp_path / "data.csv"
    plot_path = tmp_path / "plot.png"
    df = generate(
        model="cphm",
        n=5,
        beta=0.5,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=0.5,
        seed=123,
    ).rename(columns={"time": "T", "status": "E"})
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "visualize",
            str(csv_path),
            "--time-col",
            "T",
            "--status-col",
            "E",
            "--output",
            str(plot_path),
        ],
    )
    assert result.exit_code == 0
    assert plot_path.exists()


def test_visualize_cli_group_column(tmp_path) -> None:
    """CLI generates stratified plots when group column is provided."""
    csv_path = tmp_path / "data.csv"
    plot_path = tmp_path / "plot.png"
    df = generate(
        model="cphm",
        n=6,
        beta=0.5,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=0.5,
        seed=42,
    )
    df["group"] = (df["X0"] > df["X0"].median()).astype(int)
    df.to_csv(csv_path, index=False)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "visualize",
            str(csv_path),
            "--group-col",
            "group",
            "--output",
            str(plot_path),
        ],
    )
    assert result.exit_code == 0
    assert plot_path.exists()


def test_visualize_cli_default_output(tmp_path) -> None:
    """CLI writes to the default output path when none is specified."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        df = generate(
            model="cphm",
            n=4,
            beta=0.5,
            covariate_range=1.0,
            model_cens="uniform",
            cens_par=0.5,
            seed=99,
        )
        csv_path = Path("data.csv")
        df.to_csv(csv_path, index=False)
        result = runner.invoke(app, ["visualize", str(csv_path)])
        assert result.exit_code == 0
        assert Path("survival_plot.png").exists()
