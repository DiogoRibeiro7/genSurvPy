import sys
import types

import numpy as np
import pandas as pd
import pytest

from gen_surv.integration import to_sksurv
from gen_surv.interface import generate


def test_to_sksurv():
    """Basic conversion with default column names."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    arr = to_sksurv(df)
    assert arr.dtype.names == ("status", "time")
    assert arr.shape[0] == 2


def test_to_sksurv_custom_columns():
    """Unit test for custom time/event column names."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})
    arr = to_sksurv(df, time_col="T", event_col="E")
    assert arr.dtype.names == ("E", "T")


def test_to_sksurv_missing_dependency(monkeypatch):
    """Regression test ensuring a helpful ImportError is raised."""
    fake_mod = types.ModuleType("sksurv")
    monkeypatch.setitem(sys.modules, "sksurv", fake_mod)
    monkeypatch.delitem(sys.modules, "sksurv.util", raising=False)
    df = pd.DataFrame({"time": [1.0], "status": [1]})
    with pytest.raises(ImportError, match="scikit-survival is required"):
        to_sksurv(df)


def test_to_sksurv_missing_columns():
    """Regression test: missing required columns should raise KeyError."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"status": [1, 0]})
    with pytest.raises(KeyError):
        to_sksurv(df)


def test_to_sksurv_empty_dataframe():
    """Unit test for handling empty DataFrames."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [], "status": []})
    arr = to_sksurv(df)
    assert arr.shape == (0,)
    assert arr.dtype.names == ("status", "time")
    assert arr.dtype["status"] == np.dtype(bool)


def test_to_sksurv_event_dtype_non_empty():
    """Status column is coerced to boolean for non-empty inputs."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    arr = to_sksurv(df)
    assert arr.dtype["status"] == np.dtype(bool)


def test_to_sksurv_casts_float_events():
    """Float event indicators are cast to their boolean equivalents."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1.0, 0.0]})
    arr = to_sksurv(df)
    assert arr.dtype["status"] == np.dtype(bool)
    assert arr["status"].tolist() == [True, False]


def test_generate_to_sksurv_pipeline():
    """Integration test covering generation and conversion."""
    pytest.importorskip("sksurv.util")
    df = generate(
        model="cphm",
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
        seed=0,
    )
    arr = to_sksurv(df)
    assert arr.shape[0] == 5
    assert arr.dtype["status"] == np.dtype(bool)


def test_to_sksurv_rejects_non_binary_events():
    """Regression test: event column must contain only 0/1 values."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [0, 2]})
    with pytest.raises(ValueError, match="event indicator must be binary"):
        to_sksurv(df)


def test_to_sksurv_rejects_missing_events():
    """Regression test: missing event indicators trigger an error."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, None]})
    with pytest.raises(ValueError, match="event indicator contains missing values"):
        to_sksurv(df)


def test_to_sksurv_ignores_extra_columns():
    """Regression test: additional columns are ignored."""
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0], "extra": [5.0, 6.0]})
    arr = to_sksurv(df)
    assert arr.dtype.names == ("status", "time")
    assert arr.shape[0] == 2
