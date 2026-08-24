from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

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
            # A missing baseline used to skip, which meant the regression suite
            # protected nothing while still reporting success. Fail instead, so
            # an absent or unregistered baseline is visible.
            raise AssertionError(
                f"Missing baseline {path}. Regenerate it with "
                "`pytest --update-baselines` and commit the file; a baseline "
                "that is absent cannot detect a regression."
            )
        return pd.read_parquet(path)

    return _load


def assert_frame_numeric_equal(
    got: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Compare two frames column by column, numerically where possible.

    The tolerance is deliberately near machine precision. These comparisons
    back a reproducibility guarantee -- the same seed and version giving the
    same numbers -- so anything a sampler actually changes should fail, and only
    last-bit differences from a library update should pass. At the previous
    ``rtol=1e-6`` a one-part-per-million change to a hazard went undetected.
    """
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
