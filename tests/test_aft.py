"""
Tests for Accelerated Failure Time (AFT) models.
"""

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from gen_surv.aft import gen_aft_log_logistic, gen_aft_log_normal, gen_aft_weibull
from gen_surv.validation import ChoiceError, PositiveValueError


def test_gen_aft_log_logistic_runs():
    """Test that the Log-Logistic AFT generator runs without errors."""
    df = gen_aft_log_logistic(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "status" in df.columns
    assert "X0" in df.columns
    assert "X1" in df.columns
    assert set(df["status"].unique()).issubset({0, 1})


def test_gen_aft_log_logistic_invalid_shape():
    """Test that the Log-Logistic AFT generator raises error
    for invalid shape."""
    with pytest.raises(PositiveValueError):
        gen_aft_log_logistic(
            n=10,
            beta=[0.5, -0.2],
            shape=-1.0,  # Invalid negative shape
            scale=2.0,
            model_cens="uniform",
            cens_par=5.0,
        )


def test_gen_aft_log_logistic_invalid_scale():
    """Test that the Log-Logistic AFT generator raises error
    for invalid scale."""
    with pytest.raises(PositiveValueError):
        gen_aft_log_logistic(
            n=10,
            beta=[0.5, -0.2],
            shape=1.5,
            scale=0.0,  # Invalid zero scale
            model_cens="uniform",
            cens_par=5.0,
        )


@given(
    n=st.integers(min_value=1, max_value=20),
    shape=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    cens_par=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_gen_aft_log_logistic_properties(n, shape, scale, cens_par, seed):
    """Property-based test for the Log-Logistic AFT generator."""
    df = gen_aft_log_logistic(
        n=n,
        beta=[0.5, -0.2],
        shape=shape,
        scale=scale,
        model_cens="uniform",
        cens_par=cens_par,
        seed=seed,
    )
    assert df.shape[0] == n
    assert set(df["status"].unique()).issubset({0, 1})
    assert (df["time"] >= 0).all()
    assert df.filter(regex="^X[0-9]+$").shape[1] == 2


def test_gen_aft_log_logistic_reproducibility():
    """Test that the Log-Logistic AFT generator is reproducible
    with the same seed."""
    df1 = gen_aft_log_logistic(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    )

    df2 = gen_aft_log_logistic(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    )

    pd.testing.assert_frame_equal(df1, df2)

    df3 = gen_aft_log_logistic(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=43,  # Different seed
    )

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)


def test_gen_aft_log_normal_runs():
    """Test that the log-normal AFT generator runs without errors."""
    df = gen_aft_log_normal(
        n=10, beta=[0.5, -0.2], sigma=1.0, model_cens="uniform", cens_par=5.0, seed=42
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "status" in df.columns
    assert "X0" in df.columns
    assert "X1" in df.columns
    assert set(df["status"].unique()).issubset({0, 1})


def test_gen_aft_weibull_runs():
    """Test that the Weibull AFT generator runs without errors."""
    df = gen_aft_weibull(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "status" in df.columns
    assert "X0" in df.columns
    assert "X1" in df.columns
    assert set(df["status"].unique()).issubset({0, 1})


def test_gen_aft_weibull_invalid_shape():
    """Test that the Weibull AFT generator raises error for invalid shape."""
    with pytest.raises(PositiveValueError):
        gen_aft_weibull(
            n=10,
            beta=[0.5, -0.2],
            shape=-1.0,  # Invalid negative shape
            scale=2.0,
            model_cens="uniform",
            cens_par=5.0,
        )


def test_gen_aft_weibull_invalid_scale():
    """Test that the Weibull AFT generator raises error for invalid scale."""
    with pytest.raises(PositiveValueError):
        gen_aft_weibull(
            n=10,
            beta=[0.5, -0.2],
            shape=1.5,
            scale=0.0,  # Invalid zero scale
            model_cens="uniform",
            cens_par=5.0,
        )


def test_gen_aft_weibull_invalid_cens_model():
    """Test that the Weibull AFT generator raises error for invalid censoring model."""
    with pytest.raises(ChoiceError):
        gen_aft_weibull(
            n=10,
            beta=[0.5, -0.2],
            shape=1.5,
            scale=2.0,
            model_cens="invalid",  # Invalid censoring model
            cens_par=5.0,
        )


@given(
    n=st.integers(min_value=1, max_value=20),
    shape=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False
    ),
    cens_par=st.floats(
        min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_gen_aft_weibull_properties(n, shape, scale, cens_par, seed):
    """Property-based test for the Weibull AFT generator."""
    df = gen_aft_weibull(
        n=n,
        beta=[0.5, -0.2],
        shape=shape,
        scale=scale,
        model_cens="uniform",
        cens_par=cens_par,
        seed=seed,
    )
    assert df.shape[0] == n
    assert set(df["status"].unique()).issubset({0, 1})
    assert (df["time"] >= 0).all()
    assert df.filter(regex="^X[0-9]+$").shape[1] == 2


def test_gen_aft_weibull_reproducibility():
    """Test that the Weibull AFT generator is reproducible with the same seed."""
    df1 = gen_aft_weibull(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    )

    df2 = gen_aft_weibull(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    )

    pd.testing.assert_frame_equal(df1, df2)

    df3 = gen_aft_weibull(
        n=10,
        beta=[0.5, -0.2],
        shape=1.5,
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=43,  # Different seed
    )

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)
