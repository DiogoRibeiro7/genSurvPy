from __future__ import annotations

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
