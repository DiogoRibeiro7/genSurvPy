import pandas as pd
from typer.testing import CliRunner

from gen_surv.cli import app


def test_dataset_cli_integration(tmp_path):
    """Run dataset command end-to-end and verify CSV output."""
    runner = CliRunner()
    out_file = tmp_path / "data.csv"
    result = runner.invoke(
        app,
        [
            "dataset",
            "cphm",
            "--n",
            "3",
            "--beta",
            "0.5",
            "--covariate-range",
            "1.0",
            "-o",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 3
    assert {"time", "status"}.issubset(df.columns)
