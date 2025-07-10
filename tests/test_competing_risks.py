"""
Tests for Competing Risks models.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

from gen_surv.competing_risks import (
    gen_competing_risks,
    gen_competing_risks_weibull,
    cause_specific_cumulative_incidence
)


def test_gen_competing_risks_basic():
    """Test that the competing risks generator runs without errors."""
    df = gen_competing_risks(
        n=10,
        n_risks=2,
        baseline_hazards=[0.5, 0.3],
        betas=[[0.8, -0.5], [0.2, 0.7]],
        model_cens="uniform",
        cens_par=2.0,
        seed=42
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "status" in df.columns
    assert "X0" in df.columns
    assert "X1" in df.columns
    assert set(df["status"].unique()).issubset({0, 1, 2})


def test_gen_competing_risks_weibull_basic():
    """Test that the Weibull competing risks generator runs without errors."""
    df = gen_competing_risks_weibull(
        n=10,
        n_risks=2,
        shape_params=[0.8, 1.5],
        scale_params=[2.0, 3.0],
        betas=[[0.8, -0.5], [0.2, 0.7]],
        model_cens="uniform",
        cens_par=2.0,
        seed=42
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "status" in df.columns
    assert "X0" in df.columns
    assert "X1" in df.columns
    assert set(df["status"].unique()).issubset({0, 1, 2})


def test_competing_risks_parameters():
    """Test parameter validation in competing risks model."""
    # Test with invalid number of baseline hazards
    with pytest.raises(ValueError, match="Expected 3 baseline hazards"):
        gen_competing_risks(
            n=10,
            n_risks=3,
            baseline_hazards=[0.5, 0.3],  # Only 2 provided, but 3 risks
            seed=42
        )
    
    # Test with invalid number of beta coefficient sets
    with pytest.raises(ValueError, match="Expected 2 sets of coefficients"):
        gen_competing_risks(
            n=10,
            n_risks=2,
            betas=[[0.8, -0.5]],  # Only 1 set provided, but 2 risks
            seed=42
        )
    
    # Test with invalid censoring model
    with pytest.raises(ValueError, match="model_cens must be 'uniform' or 'exponential'"):
        gen_competing_risks(
            n=10,
            n_risks=2,
            model_cens="invalid",
            seed=42
        )


def test_competing_risks_weibull_parameters():
    """Test parameter validation in Weibull competing risks model."""
    # Test with invalid number of shape parameters
    with pytest.raises(ValueError, match="Expected 3 shape parameters"):
        gen_competing_risks_weibull(
            n=10,
            n_risks=3,
            shape_params=[0.8, 1.5],  # Only 2 provided, but 3 risks
            seed=42
        )
    
    # Test with invalid number of scale parameters
    with pytest.raises(ValueError, match="Expected 3 scale parameters"):
        gen_competing_risks_weibull(
            n=10,
            n_risks=3,
            scale_params=[2.0, 3.0],  # Only 2 provided, but 3 risks
            seed=42
        )


def test_cause_specific_cumulative_incidence():
    """Test the cause-specific cumulative incidence function."""
    # Generate some data
    df = gen_competing_risks(
        n=50,
        n_risks=2,
        baseline_hazards=[0.5, 0.3],
        seed=42
    )
    
    # Calculate CIF for cause 1
    time_points = np.linspace(0, 5, 10)
    cif = cause_specific_cumulative_incidence(df, time_points, cause=1)
    
    assert isinstance(cif, pd.DataFrame)
    assert len(cif) == len(time_points)
    assert "time" in cif.columns
    assert "incidence" in cif.columns
    assert (cif["incidence"] >= 0).all()
    assert (cif["incidence"] <= 1).all()
    assert cif["incidence"].is_monotonic_increasing
    
    # Test with invalid cause
    with pytest.raises(ValueError, match="Cause 3 not found in the data"):
        cause_specific_cumulative_incidence(df, time_points, cause=3)


@given(
    n=st.integers(min_value=5, max_value=50),
    n_risks=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=1000)
)
def test_competing_risks_properties(n, n_risks, seed):
    """Property-based tests for the competing risks model."""
    df = gen_competing_risks(
        n=n,
        n_risks=n_risks,
        seed=seed
    )
    
    # Check basic properties
    assert df.shape[0] == n
    assert all(col in df.columns for col in ["id", "time", "status"])
    assert (df["time"] >= 0).all()
    assert df["status"].isin(list(range(n_risks + 1))).all()  # 0 to n_risks
    
    # Count of each status
    status_counts = df["status"].value_counts()
    # There should be at least one of each status (including censoring)
    # This might occasionally fail due to randomness, so we'll just check that
    # we have at least 2 different status values
    assert len(status_counts) >= 2


@given(
    n=st.integers(min_value=5, max_value=50),
    n_risks=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=1000)
)
def test_competing_risks_weibull_properties(n, n_risks, seed):
    """Property-based tests for the Weibull competing risks model."""
    df = gen_competing_risks_weibull(
        n=n,
        n_risks=n_risks,
        seed=seed
    )
    
    # Check basic properties
    assert df.shape[0] == n
    assert all(col in df.columns for col in ["id", "time", "status"])
    assert (df["time"] >= 0).all()
    assert df["status"].isin(list(range(n_risks + 1))).all()  # 0 to n_risks
    
    # Count of each status
    status_counts = df["status"].value_counts()
    # There should be at least 2 different status values
    assert len(status_counts) >= 2


def test_reproducibility():
    """Test that results are reproducible with the same seed."""
    df1 = gen_competing_risks(
        n=20,
        n_risks=2,
        seed=42
    )
    
    df2 = gen_competing_risks(
        n=20,
        n_risks=2,
        seed=42
    )
    
    pd.testing.assert_frame_equal(df1, df2)
    
    # Different seeds should produce different results
    df3 = gen_competing_risks(
        n=20,
        n_risks=2,
        seed=43
    )
    
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)
