from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

BASELINE_DIR = Path(__file__).parent / "baselines"
BASELINE_DIR.mkdir(exist_ok=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Refresh stored baselines for regression tests.",
    )


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def save_baseline() -> Callable[[pd.DataFrame, str], None]:
    def _save(df: pd.DataFrame, name: str) -> None:
        (BASELINE_DIR / f"{name}.parquet").write_bytes(df.to_parquet(index=False))

    return _save


@pytest.fixture(scope="session")
def load_baseline() -> Callable[[str], pd.DataFrame]:
    def _load(name: str) -> pd.DataFrame:
        path = BASELINE_DIR / f"{name}.parquet"
        if not path.exists():
            pytest.skip(
                f"Missing baseline {path}; run with --update-baselines to refresh."
            )
        return pd.read_parquet(path)

    return _load


def assert_frame_numeric_equal(
    got: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    assert list(got.columns) == list(expected.columns), "Column order/name changed."
    assert got.shape == expected.shape, "Shape changed."
    for col in got.columns:
        g = pd.to_numeric(got[col], errors="coerce")
        e = pd.to_numeric(expected[col], errors="coerce")
        if g.notna().all() and e.notna().all():
            np.testing.assert_allclose(g.to_numpy(), e.to_numpy(), rtol=rtol, atol=atol)
        else:
            assert (
                got[col].astype(str).values == expected[col].astype(str).values
            ).all(), f"Mismatch in column {col!r}"
