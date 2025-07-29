"""Data export utilities for gen_surv.

This module provides helper functions to save generated
survival datasets in various formats.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd


def export_dataset(df: pd.DataFrame, path: str, fmt: Optional[str] = None) -> None:
    """Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data.
    path : str
        File path to write to. The extension is used to infer the format
        when ``fmt`` is ``None``.
    fmt : {"csv", "json", "feather"}, optional
        Format to use. If omitted, inferred from ``path``.

    Raises
    ------
    ValueError
        If the format is not one of the supported types.
    """
    if fmt is None:
        fmt = os.path.splitext(path)[1].lstrip(".").lower()

    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="table")
    elif fmt in {"feather", "ft"}:
        df.reset_index(drop=True).to_feather(path)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
