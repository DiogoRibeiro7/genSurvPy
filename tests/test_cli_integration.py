import pandas as pd
import pytest
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


# Arguments each model needs beyond --n and --seed. A model absent from here is
# generatable from the defaults alone.
CLI_ARGUMENTS = {
    "aft_ln": ["--beta", "0.5", "--beta", "-0.3", "--sigma", "1.0"],
    "aft_weibull": [
        "--beta",
        "0.5",
        "--beta",
        "-0.3",
        "--shape",
        "1.5",
        "--scale",
        "2.0",
    ],
    "aft_log_logistic": [
        "--beta",
        "0.5",
        "--beta",
        "-0.3",
        "--shape",
        "1.5",
        "--scale",
        "2.0",
    ],
    "mixture_cure": ["--cure-fraction", "0.3"],
    "piecewise_exponential": [
        "--breakpoints",
        "1.0",
        "--hazard-rates",
        "0.5",
        "--hazard-rates",
        "1.5",
    ],
    "cmm": [
        "--beta",
        "0.1",
        "--beta",
        "0.2",
        "--beta",
        "0.3",
        "--rate",
        "0.1",
        "--rate",
        "1.0",
        "--rate",
        "0.2",
        "--rate",
        "1.0",
        "--rate",
        "0.1",
        "--rate",
        "1.0",
    ],
    "thmm": [
        "--beta",
        "0.1",
        "--beta",
        "0.2",
        "--beta",
        "0.3",
        "--rate",
        "0.2",
        "--rate",
        "0.3",
        "--rate",
        "0.4",
    ],
    "tdcm": [
        "--beta",
        "0.5",
        "--beta",
        "0.3",
        "--dist",
        "weibull",
        "--corr",
        "0.5",
        "--lam",
        "1.0",
    ],
    "recurrent_events": ["--baseline", "exponential", "--rate", "0.5"],
}


def _registered_models():
    from gen_surv.interface import _model_map

    return sorted(_model_map)


@pytest.mark.parametrize("model", _registered_models())
def test_every_model_is_generatable_from_the_command_line(model, tmp_path):
    """The CLI offers every registered model, so every one must actually run.

    Before the rate and tdcm options existed, `cmm`, `thmm` and `tdcm` failed
    with a TypeError about missing positional arguments -- the CLI listed them
    as valid values of MODEL while having no way to supply their parameters.
    """
    out_file = tmp_path / f"{model}.csv"
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            model,
            "--n",
            "5",
            "--seed",
            "1",
            *CLI_ARGUMENTS.get(model, []),
            "-o",
            str(out_file),
        ],
    )

    assert result.exit_code == 0, f"{model} failed: {result.output}"
    assert out_file.exists()
    assert len(pd.read_csv(out_file)) > 0
