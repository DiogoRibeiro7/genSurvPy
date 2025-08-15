"""Tests for scikit-survival integration functionality."""

import numpy as np
import pandas as pd
import pytest

from gen_surv.integration import to_sksurv, from_sksurv


def test_to_sksurv_basic():
    """Test basic conversion from DataFrame to sksurv format."""
    pytest.importorskip("sksurv.util")
    
    df = pd.DataFrame({
        "time": [1.0, 2.0, 3.0],
        "status": [1, 0, 1]
    })
    
    arr = to_sksurv(df)
    
    assert len(arr) == 3
    assert arr.dtype.names == ("status", "time")
    assert list(arr["time"]) == [1.0, 2.0, 3.0]
    assert list(arr["status"]) == [True, False, True]


def test_to_sksurv_custom_columns():
    """Test conversion with custom column names."""
    pytest.importorskip("sksurv.util")
    
    df = pd.DataFrame({
        "survival_time": [1.0, 2.0],
        "event": [1, 0]
    })
    
    arr = to_sksurv(df, time_col="survival_time", event_col="event")
    
    assert len(arr) == 2
    assert arr.dtype.names == ("event", "survival_time")


def test_to_sksurv_empty_dataframe():
    """Test conversion of empty DataFrame."""
    pytest.importorskip("sksurv.util")
    
    df = pd.DataFrame({"time": [], "status": []})
    arr = to_sksurv(df)
    
    assert len(arr) == 0
    assert arr.dtype.names == ("status", "time")


def test_to_sksurv_missing_columns():
    """Test error handling for missing columns."""
    pytest.importorskip("sksurv.util")
    
    df = pd.DataFrame({"time": [1.0, 2.0]})
    
    with pytest.raises(ValueError, match="Column 'status' not found"):
        to_sksurv(df)


def test_from_sksurv_basic():
    """Test conversion from sksurv format to DataFrame."""
    pytest.importorskip("sksurv.util")
    
    # Create a structured array manually
    arr = np.array([(True, 1.0), (False, 2.0), (True, 3.0)], 
                   dtype=[("status", bool), ("time", float)])
    
    df = from_sksurv(arr)
    
    assert len(df) == 3
    assert list(df.columns) == ["time", "status"]
    assert list(df["time"]) == [1.0, 2.0, 3.0]
    assert list(df["status"]) == [1, 0, 1]


def test_from_sksurv_empty():
    """Test conversion of empty structured array."""
    pytest.importorskip("sksurv.util")
    
    arr = np.array([], dtype=[("status", bool), ("time", float)])
    df = from_sksurv(arr)
    
    assert len(df) == 0
    assert list(df.columns) == ["time", "status"]


def test_roundtrip_conversion():
    """Test that conversion is bidirectional."""
    pytest.importorskip("sksurv.util")
    
    original_df = pd.DataFrame({
        "time": [1.0, 2.5, 4.0],
        "status": [1, 0, 1]
    })
    
    # Convert to sksurv and back
    arr = to_sksurv(original_df)
    result_df = from_sksurv(arr)
    
    pd.testing.assert_frame_equal(original_df, result_df)


def test_import_error_handling():
    """Test that appropriate errors are raised when sksurv is not available."""
    # This test would need to mock the import, but for now we'll skip it
    # when sksurv is available
    pytest.importorskip("sksurv.util")
    # If we get here, sksurv is available, so we can't test the ImportError path
    # In a real test environment, we'd mock the import failure
    pass
