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

    return Surv.from_dataframe(event_col, time_col, df)
