import pandas as pd
import pyreadr
import pytest

from gen_surv._validation import ChoiceError
from gen_surv.export import export_dataset


@pytest.mark.parametrize(
    "fmt, reader",
    [
        ("csv", pd.read_csv),
        ("feather", pd.read_feather),
        ("ft", pd.read_feather),
    ],
)
def test_export_dataset_formats(fmt, reader, tmp_path):
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    out = tmp_path / f"data.{fmt}"
    export_dataset(df, out)
    assert out.exists()
    result = reader(out).astype(df.dtypes.to_dict())
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df)


def test_export_dataset_json(monkeypatch, tmp_path):
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    out = tmp_path / "data.json"

    called = {}

    def fake_to_json(self, path, orient="table"):
        called["args"] = (path, orient)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")

    monkeypatch.setattr(pd.DataFrame, "to_json", fake_to_json)
    export_dataset(df, out)
    assert called["args"] == (out, "table")
    assert out.exists()


def test_export_dataset_rds(monkeypatch, tmp_path):
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    out = tmp_path / "data.rds"

    captured = {}

    def fake_write_rds(path, data):
        captured["path"] = path
        captured["data"] = data
        open(path, "wb").close()

    monkeypatch.setattr(pyreadr, "write_rds", fake_write_rds)
    export_dataset(df, out)
    assert out.exists()
    pd.testing.assert_frame_equal(captured["data"], df.reset_index(drop=True))


def test_export_dataset_explicit_fmt(monkeypatch, tmp_path):
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    out = tmp_path / "data.bin"

    called = {}

    def fake_to_json(self, path, orient="table"):
        called["args"] = (path, orient)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")

    monkeypatch.setattr(pd.DataFrame, "to_json", fake_to_json)
    export_dataset(df, out, fmt="json")
    assert called["args"] == (out, "table")
    assert out.exists()


def test_export_dataset_invalid_format(tmp_path):
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    with pytest.raises(ChoiceError):
        export_dataset(df, tmp_path / "data.xxx", fmt="txt")
