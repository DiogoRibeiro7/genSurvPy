"""Integration utilities for interfacing with scikit-survival."""

import numpy as np
import pandas as pd

try:
    from sksurv.util import Surv

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False


def to_sksurv(
    df: pd.DataFrame, time_col: str = "time", event_col: str = "status"
) -> np.ndarray:
    """
    Convert a pandas DataFrame to a scikit-survival structured array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data
    time_col : str, default "time"
        Name of the column containing survival times
    event_col : str, default "status"
        Name of the column containing event indicators (0/1 or boolean)

    Returns
    -------
    y : structured array
        Structured array suitable for scikit-survival functions

    Raises
    ------
    ImportError
        If scikit-survival is not installed
    ValueError
        If the DataFrame is empty or columns are missing
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is required but not installed")

    if df.empty:
        # Handle empty DataFrame case by creating a minimal valid structured array
        # This avoids the "event indicator must be binary" error for empty arrays
        return np.array([], dtype=[(event_col, bool), (time_col, float)])

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")
    if event_col not in df.columns:
        raise ValueError(f"Column '{event_col}' not found in DataFrame")

    return Surv.from_dataframe(event_col, time_col, df)


def from_sksurv(
    y: np.ndarray, time_col: str = "time", event_col: str = "status"
) -> pd.DataFrame:
    """
    Convert a scikit-survival structured array to a pandas DataFrame.

    Parameters
    ----------
    y : structured array
        Structured array from scikit-survival
    time_col : str, default "time"
        Name for the time column in the resulting DataFrame
    event_col : str, default "status"
        Name for the event column in the resulting DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with time and event columns
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is required but not installed")

    if len(y) == 0:
        return pd.DataFrame({time_col: [], event_col: []})

    # Extract field names from structured array
    event_field = y.dtype.names[0]
    time_field = y.dtype.names[1]

    return pd.DataFrame(
        {time_col: y[time_field], event_col: y[event_field].astype(int)}
    )
