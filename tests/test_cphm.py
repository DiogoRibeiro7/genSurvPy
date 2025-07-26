"""
Tests for the Cox Proportional Hazards Model (CPHM) generator.
"""

import pytest
import pandas as pd
from gen_surv.cphm import gen_cphm


def test_gen_cphm_output_shape():
    """Test that the output DataFrame has the expected shape and columns."""
    df = gen_cphm(n=50, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0)
    assert df.shape == (50, 3)
    assert list(df.columns) == ["time", "status", "X0"]


def test_gen_cphm_status_range():
    """Test that status values are binary (0 or 1)."""
    df = gen_cphm(n=100, model_cens="exponential", cens_par=0.8, beta=0.3, covariate_range=1.5)
    assert df["status"].isin([0, 1]).all()


def test_gen_cphm_time_positive():
    """Test that all time values are positive."""
    df = gen_cphm(n=50, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0)
    assert (df["time"] > 0).all()


def test_gen_cphm_covariate_range():
    """Test that covariate values are within the specified range."""
    covar_max = 2.5
    df = gen_cphm(n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=covar_max)
    assert (df["X0"] >= 0).all()
    assert (df["X0"] <= covar_max).all()


def test_gen_cphm_seed_reproducibility():
    """Test that setting the same seed produces identical results."""
    df1 = gen_cphm(n=10, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0, seed=42)
    df2 = gen_cphm(n=10, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_gen_cphm_different_seeds():
    """Test that different seeds produce different results."""
    df1 = gen_cphm(n=10, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0, seed=42)
    df2 = gen_cphm(n=10, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0, seed=43)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df2)
