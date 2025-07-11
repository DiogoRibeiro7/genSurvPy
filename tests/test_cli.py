import sys
import os
import runpy

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gen_surv.cli import dataset


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
