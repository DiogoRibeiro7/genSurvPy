from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def to_sksurv(
    df: pd.DataFrame, time_col: str = "time", event_col: str = "status"
) -> NDArray[np.void]:
    """Convert a DataFrame to a scikit-survival structured array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data.
    time_col : str, default "time"
        Column storing durations.
    event_col : str, default "status"
        Column storing event indicators (1=event, 0=censored).

    Returns
    -------
    numpy.ndarray
        Structured array suitable for scikit-survival estimators.

    Notes
    -----
    The ``sksurv`` package is imported lazily inside the function. It must be
    installed separately, for instance with ``pip install scikit-survival``.
    """

    try:
        from sksurv.util import Surv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-survival is required for this feature.") from exc

    # ``Surv.from_dataframe`` expects the event indicator to be boolean.
    # Validate the column is binary before casting to avoid silently
    # accepting unexpected values (e.g., NaNs or numbers other than 0/1).
    df_copy = df.copy()
    events = df_copy[event_col]
    if events.isna().any():
        raise ValueError("event indicator contains missing values")
    if not events.isin([0, 1, False, True]).all():
        raise ValueError("event indicator must be binary")
    df_copy[event_col] = events.astype(bool)

    return Surv.from_dataframe(event_col, time_col, df_copy)
