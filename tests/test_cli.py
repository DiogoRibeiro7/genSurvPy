import os
import runpy
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gen_surv.cli import dataset, visualize


def test_cli_dataset_stdout(monkeypatch, capsys):
    """
    Test that the 'dataset' CLI command prints the generated CSV data to stdout when no output file is specified.
    This test patches the 'generate' function to return a simple DataFrame, invokes the CLI command directly,
    and asserts that the expected CSV header appears in the captured standard output.
    """

    def fake_generate(model: str, n: int):
        return pd.DataFrame({"time": [1.0], "status": [1], "X0": [0.1], "X1": [0.2]})

    # Patch the generate function used in the CLI to avoid heavy computation.
    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    # Call the command function directly to sidestep Click argument parsing
    dataset(model="cphm", n=1, output=None)
    captured = capsys.readouterr()
    assert "time,status,X0,X1" in captured.out


def test_main_entry_point(monkeypatch):
    """Running the module as a script should invoke the CLI app."""

    called = []

    def fake_app():
        called.append(True)

    # Patch the CLI app before the module is executed
    monkeypatch.setattr("gen_surv.cli.app", fake_app)
    monkeypatch.setattr("sys.argv", ["gen_surv", "dataset", "cphm"])
    runpy.run_module("gen_surv.__main__", run_name="__main__")
    assert called


def test_cli_dataset_file_output(monkeypatch, tmp_path):
    """Dataset command writes CSV to file when output path is provided."""

    def fake_generate(model: str, n: int):
        return pd.DataFrame({"time": [1.0], "status": [1], "X0": [0.1], "X1": [0.2]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    out_file = tmp_path / "out.csv"
    dataset(model="cphm", n=1, output=str(out_file))
    assert out_file.exists()
    content = out_file.read_text()
    assert "time,status,X0,X1" in content


def test_dataset_fallback(monkeypatch):
    """If generate fails with additional kwargs, dataset retries with minimal args."""
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TypeError("bad args")
        return pd.DataFrame({"time": [0], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(model="cphm", n=2, output=None)
    # first call has many parameters, second only model and n
    assert calls[-1] == {"model": "cphm", "n": 2}
    assert len(calls) == 2


def test_dataset_weibull_parameters(monkeypatch):
    """Parameters for aft_weibull model are forwarded correctly."""
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [0]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(
        model="aft_weibull", n=3, beta=[0.1, 0.2], shape=1.1, scale=2.2, output=None
    )
    assert captured["model"] == "aft_weibull"
    assert captured["beta"] == [0.1, 0.2]
    assert captured["shape"] == 1.1
    assert captured["scale"] == 2.2


def test_dataset_aft_ln(monkeypatch):
    """aft_ln model should forward beta list and sigma."""
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(model="aft_ln", n=1, beta=[0.3, 0.4], sigma=1.2, output=None)
    assert captured["beta"] == [0.3, 0.4]
    assert captured["sigma"] == 1.2


def test_dataset_competing_risks(monkeypatch):
    """competing_risks expands betas and passes hazards."""
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(
        model="competing_risks",
        n=1,
        n_risks=2,
        baseline_hazards=[0.1, 0.2],
        beta=0.5,
        output=None,
    )
    assert captured["n_risks"] == 2
    assert captured["baseline_hazards"] == [0.1, 0.2]
    assert captured["betas"] == [0.5, 0.5]


def test_dataset_mixture_cure(monkeypatch):
    """mixture_cure passes cure and baseline parameters."""
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(
        model="mixture_cure",
        n=1,
        cure_fraction=0.2,
        baseline_hazard=0.1,
        beta=[0.4],
        output=None,
    )
    assert captured["cure_fraction"] == 0.2
    assert captured["baseline_hazard"] == 0.1
    assert captured["betas_survival"] == [0.4]
    assert captured["betas_cure"] == [0.4]


def test_dataset_invalid_model(monkeypatch):
    def fake_generate(**kwargs):
        raise ValueError("bad model")

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    with pytest.raises(ValueError):
        dataset(model="nope", n=1, output=None)


def test_cli_visualize_basic(monkeypatch, tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"time": [1, 2], "status": [1, 0]}).to_csv(csv, index=False)

    def fake_plot_survival_curve(**kwargs):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        return fig, ax

    monkeypatch.setattr(
        "gen_surv.visualization.plot_survival_curve", fake_plot_survival_curve
    )

    saved = []

    def fake_savefig(path, *args, **kwargs):
        saved.append(path)

    monkeypatch.setattr("matplotlib.pyplot.savefig", fake_savefig)

    visualize(
        str(csv),
        time_col="time",
        status_col="status",
        group_col=None,
        output=str(tmp_path / "plot.png"),
    )
    assert saved and saved[0].endswith("plot.png")


def test_dataset_aft_log_logistic(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(
        model="aft_log_logistic",
        n=1,
        beta=[0.1],
        shape=1.2,
        scale=2.3,
        output=None,
    )
    assert captured["model"] == "aft_log_logistic"
    assert captured["beta"] == [0.1]
    assert captured["shape"] == 1.2
    assert captured["scale"] == 2.3


def test_dataset_competing_risks_weibull(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(
        model="competing_risks_weibull",
        n=1,
        n_risks=2,
        shape_params=[0.7, 1.2],
        scale_params=[2.0, 2.0],
        beta=0.3,
        output=None,
    )
    assert captured["n_risks"] == 2
    assert captured["shape_params"] == [0.7, 1.2]
    assert captured["scale_params"] == [2.0, 2.0]
    assert captured["betas"] == [0.3, 0.3]


def test_dataset_piecewise(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"time": [1], "status": [1]})

    monkeypatch.setattr("gen_surv.cli.generate", fake_generate)
    dataset(
        model="piecewise_exponential",
        n=1,
        breakpoints=[1.0],
        hazard_rates=[0.2, 0.3],
        beta=[0.4],
        output=None,
    )
    assert captured["breakpoints"] == [1.0]
    assert captured["hazard_rates"] == [0.2, 0.3]
    assert captured["betas"] == [0.4]
