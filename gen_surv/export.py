"""Data export utilities for gen_surv.

This module provides helper functions to save generated
survival datasets in various formats.
"""

from __future__ import annotations

import os

import pandas as pd
from dataexcept import FileWriteError

from .validation import ensure_in_choices


def export_dataset(
    df: pd.DataFrame, path: str | os.PathLike[str], fmt: str | None = None
) -> None:
    """Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data.
    path : str or os.PathLike[str]
        File path to write to. The extension is used to infer the format
        when ``fmt`` is ``None``.
    fmt : {"csv", "json", "feather", "rds"}, optional
        Format to use. If omitted, inferred from ``path``.

    Raises
    ------
    ChoiceError
        If the format is not one of the supported types.
    FileWriteError
        If writing the dataset fails. The original ``OSError`` is chained.
    """
    if fmt is None:
        fmt = os.path.splitext(path)[1].lstrip(".").lower()

    ensure_in_choices(fmt, "fmt", {"csv", "json", "feather", "ft", "rds"})

    try:
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="table")
        elif fmt in {"feather", "ft"}:
            df.reset_index(drop=True).to_feather(path)
        elif fmt == "rds":
            try:
                import pyreadr  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise ModuleNotFoundError(
                    "pyreadr is required for RDS export; install the 'pyreadr' package."
                ) from exc
            pyreadr.write_rds(path, df.reset_index(drop=True))
    except OSError as exc:
        raise FileWriteError(os.fspath(path), original=exc) from exc
