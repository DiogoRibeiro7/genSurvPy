"""Tests for Competing Risks models."""

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import gen_surv.competing_risks as cr
from gen_surv.competing_risks import (
    cause_specific_cumulative_incidence,
    gen_competing_risks,
    gen_competing_risks_weibull,
)
from gen_surv.validation import (
    ChoiceError,
    LengthError,
    NumericSequenceError,
    ParameterError,
    PositiveSequenceError,
    PositiveValueError,
)

os.environ.setdefault("MPLBACKEND", "Agg")


def test_gen_competing_risks_basic():
    """Test that the competing risks generator runs without errors."""
    df = gen_competing_risks(
        n=10,
        n_risks=2,
        baseline_hazards=[0.5, 0.3],
        betas=[[0.8, -0.5], [0.2, 0.7]],
        model_cens="uniform",
        cens_par=2.0,
        seed=42,
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
        seed=42,
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
    with pytest.raises(LengthError):
        gen_competing_risks(
            n=10,
            n_risks=3,
            baseline_hazards=[0.5, 0.3],  # Only 2 provided, but 3 risks
            seed=42,
        )

    # Test with invalid number of beta coefficient sets
    with pytest.raises(LengthError):
        gen_competing_risks(
            n=10,
            n_risks=2,
            betas=[[0.8, -0.5]],  # Only 1 set provided, but 2 risks
            seed=42,
        )

    # Test with invalid censoring model
    with pytest.raises(ChoiceError):
        gen_competing_risks(n=10, n_risks=2, model_cens="invalid", seed=42)


def test_invalid_covariate_dist():
    with pytest.raises(ChoiceError):
        gen_competing_risks(n=5, n_risks=2, covariate_dist="unknown", seed=1)
    with pytest.raises(ChoiceError):
        gen_competing_risks_weibull(n=5, n_risks=2, covariate_dist="unknown", seed=1)


def test_competing_risks_positive_params():
    with pytest.raises(PositiveValueError):
        gen_competing_risks(n=5, n_risks=2, cens_par=0.0, seed=0)
    with pytest.raises(PositiveValueError):
        gen_competing_risks(n=5, n_risks=2, max_time=-1.0, seed=0)


def test_competing_risks_invalid_covariate_params():
    with pytest.raises(ParameterError):
        gen_competing_risks(
            n=5,
            n_risks=2,
            covariate_dist="normal",
            covariate_params={"mean": 0.0},
            seed=1,
        )
    with pytest.raises(PositiveValueError):
        gen_competing_risks(
            n=5,
            n_risks=2,
            covariate_dist="normal",
            covariate_params={"mean": 0.0, "std": -1.0},
            seed=1,
        )
    with pytest.raises(ParameterError):
        gen_competing_risks_weibull(
            n=5,
            n_risks=2,
            covariate_dist="binary",
            covariate_params={"p": 1.5},
            seed=1,
        )


def test_competing_risks_invalid_beta_values():
    with pytest.raises(NumericSequenceError):
        gen_competing_risks(n=5, n_risks=2, betas=[[0.1, "x"], [0.2, 0.3]], seed=0)
    with pytest.raises(NumericSequenceError):
        gen_competing_risks_weibull(
            n=5, n_risks=2, betas=[[0.1, np.nan], [0.2, 0.3]], seed=0
        )


def test_competing_risks_weibull_parameters():
    """Test parameter validation in Weibull competing risks model."""
    # Test with invalid number of shape parameters
    with pytest.raises(LengthError):
        gen_competing_risks_weibull(
            n=10,
            n_risks=3,
            shape_params=[0.8, 1.5],  # Only 2 provided, but 3 risks
            seed=42,
        )

    # Test with invalid number of scale parameters
    with pytest.raises(LengthError):
        gen_competing_risks_weibull(
            n=10,
            n_risks=3,
            scale_params=[2.0, 3.0],  # Only 2 provided, but 3 risks
            seed=42,
        )


def test_competing_risks_weibull_positive_params():
    with pytest.raises(PositiveSequenceError):
        gen_competing_risks_weibull(n=5, n_risks=2, shape_params=[1.0, -1.0], seed=0)
    with pytest.raises(PositiveSequenceError):
        gen_competing_risks_weibull(n=5, n_risks=2, scale_params=[2.0, 0.0], seed=0)
    with pytest.raises(PositiveValueError):
        gen_competing_risks_weibull(n=5, n_risks=2, cens_par=-1.0, seed=0)
    with pytest.raises(PositiveValueError):
        gen_competing_risks_weibull(n=5, n_risks=2, max_time=0.0, seed=0)


def test_cause_specific_cumulative_incidence():
    """Test the cause-specific cumulative incidence function."""
    # Generate some data
    df = gen_competing_risks(n=50, n_risks=2, baseline_hazards=[0.5, 0.3], seed=42)

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
    with pytest.raises(ParameterError):
        cause_specific_cumulative_incidence(df, time_points, cause=3)


def test_cause_specific_cumulative_incidence_handles_ties():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "time": [1.0, 1.0, 2.0, 2.0],
            "status": [1, 2, 1, 0],
        }
    )
    cif = cause_specific_cumulative_incidence(df, [1.0, 2.0], cause=1)
    assert np.allclose(cif["incidence"].to_numpy(), [0.25, 0.5])


def test_cause_specific_cumulative_incidence_bounds():
    df = gen_competing_risks(n=30, n_risks=2, seed=5)
    max_time = df["time"].max()
    time_points = [-1.0, 0.0, max_time + 1]
    cif = cause_specific_cumulative_incidence(df, time_points, cause=1)
    assert cif.iloc[0]["incidence"] == 0.0
    expected = cause_specific_cumulative_incidence(df, [max_time], cause=1).iloc[0][
        "incidence"
    ]
    assert cif.iloc[-1]["incidence"] == expected


@given(
    n=st.integers(min_value=5, max_value=50),
    n_risks=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=1000),
)
def test_competing_risks_properties(n, n_risks, seed):
    """Property-based tests for the competing risks model."""
    df = gen_competing_risks(n=n, n_risks=n_risks, seed=seed)

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
    seed=st.integers(min_value=0, max_value=1000),
)
def test_competing_risks_weibull_properties(n, n_risks, seed):
    """Property-based tests for the Weibull competing risks model."""
    df = gen_competing_risks_weibull(n=n, n_risks=n_risks, seed=seed)

    # Check basic properties
    assert df.shape[0] == n
    assert all(col in df.columns for col in ["id", "time", "status"])
    assert (df["time"] >= 0).all()
    assert df["status"].isin(list(range(n_risks + 1))).all()  # 0 to n_risks

    # Count of each status
    status_counts = df["status"].value_counts()
    # There should be at least 2 different status values
    assert len(status_counts) >= 2


def test_gen_competing_risks_forces_event_types():
    df = gen_competing_risks(
        n=2,
        n_risks=2,
        baseline_hazards=[1e-9, 1e-9],
        model_cens="uniform",
        cens_par=0.1,
        seed=0,
    )
    assert set(df["status"]) == {1, 2}


def test_gen_competing_risks_weibull_forces_event_types():
    df = gen_competing_risks_weibull(
        n=2,
        n_risks=2,
        shape_params=[1, 1],
        scale_params=[1e9, 1e9],
        model_cens="uniform",
        cens_par=0.1,
        seed=0,
    )
    assert set(df["status"]) == {1, 2}


def test_reproducibility():
    """Test that results are reproducible with the same seed."""
    df1 = gen_competing_risks(n=20, n_risks=2, seed=42)

    df2 = gen_competing_risks(n=20, n_risks=2, seed=42)

    pd.testing.assert_frame_equal(df1, df2)

    # Different seeds should produce different results
    df3 = gen_competing_risks(n=20, n_risks=2, seed=43)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df3)


def test_competing_risks_summary_basic():
    df = gen_competing_risks(n=10, n_risks=2, seed=1)
    summary = cr.competing_risks_summary(df)
    assert summary["n_subjects"] == 10
    assert summary["n_causes"] == 2
    assert set(summary["events_by_cause"]) <= {1, 2}
    assert "time_stats" in summary


def test_competing_risks_summary_with_categorical():
    df = gen_competing_risks(n=8, n_risks=2, seed=2)
    df["group"] = ["A", "B"] * 4
    summary = cr.competing_risks_summary(df, covariate_cols=["X0", "group"])
    assert summary["covariate_stats"]["group"]["categories"] == 2
    assert "distribution" in summary["covariate_stats"]["group"]


def test_plot_cause_specific_hazards_runs():
    plt = pytest.importorskip("matplotlib.pyplot")
    df = gen_competing_risks(n=30, n_risks=2, seed=3)
    fig, ax = cr.plot_cause_specific_hazards(df, time_points=np.linspace(0, 5, 5))
    assert hasattr(fig, "savefig")
    assert len(ax.get_lines()) >= 1
    plt.close(fig)
