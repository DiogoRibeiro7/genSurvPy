import numpy as np
from gen_surv.piecewise import piecewise_hazard_function, piecewise_survival_function


def test_piecewise_hazard_function_scalar_and_array():
    breakpoints = [1.0, 2.0]
    hazard_rates = [0.5, 1.0, 1.5]
    # Scalar values
    assert piecewise_hazard_function(0.5, breakpoints, hazard_rates) == 0.5
    assert piecewise_hazard_function(1.5, breakpoints, hazard_rates) == 1.0
    assert piecewise_hazard_function(3.0, breakpoints, hazard_rates) == 1.5
    # Array values
    arr = np.array([0.5, 1.5, 3.0])
    np.testing.assert_allclose(
        piecewise_hazard_function(arr, breakpoints, hazard_rates),
        np.array([0.5, 1.0, 1.5]),
    )


def test_piecewise_hazard_function_negative_time():
    """Hazard should be zero for negative times."""
    breakpoints = [1.0, 2.0]
    hazard_rates = [0.5, 1.0, 1.5]
    assert piecewise_hazard_function(-1.0, breakpoints, hazard_rates) == 0
    np.testing.assert_array_equal(
        piecewise_hazard_function(np.array([-0.5, -2.0]), breakpoints, hazard_rates),
        np.array([0.0, 0.0]),
    )


def test_piecewise_survival_function():
    breakpoints = [1.0, 2.0]
    hazard_rates = [0.5, 1.0, 1.5]
    # Known survival probabilities
    expected = np.exp(-np.array([0.0, 0.25, 1.0, 3.0]))
    times = np.array([0.0, 0.5, 1.5, 3.0])
    np.testing.assert_allclose(
        piecewise_survival_function(times, breakpoints, hazard_rates),
        expected,
    )


def test_piecewise_survival_function_scalar_and_negative():
    breakpoints = [1.0, 2.0]
    hazard_rates = [0.5, 1.0, 1.5]
    # Scalar output should be a float
    val = piecewise_survival_function(1.5, breakpoints, hazard_rates)
    assert isinstance(val, float)
    assert np.isclose(val, np.exp(-1.0))
    # Negative times return survival of 1
    assert piecewise_survival_function(-2.0, breakpoints, hazard_rates) == 1
