"""Typed file boundaries for the study's Parquet artifacts.

These helpers classify failures of external files. Frame construction and
numerical validation remain the responsibility of their callers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dataexcept import DataLoadingError, FileWriteError


def read_parquet(path: Path | str) -> pd.DataFrame:
    """Load a Parquet artifact, retaining I/O and corrupt-file causes."""
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DataLoadingError(str(path), exc) from exc


def write_parquet(frame: pd.DataFrame, path: Path | str) -> None:
    """Write an artifact and preserve the original filesystem failure."""
    try:
        frame.to_parquet(path, index=False)
    except OSError as exc:
        raise FileWriteError(str(path), original=exc) from exc


def make_directory(path: Path) -> None:
    """Create an artifact directory, classifying filesystem failures."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FileWriteError(str(path), original=exc) from exc
