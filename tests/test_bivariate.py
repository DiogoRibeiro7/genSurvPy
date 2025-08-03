import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

from gen_surv._validation import ChoiceError, LengthError
from gen_surv.bivariate import sample_bivariate_distribution


def test_sample_bivariate_exponential_shape():
    """Exponential distribution should return an array of shape (n, 2)."""
    result = sample_bivariate_distribution(5, "exponential", 0.0, [1.0, 1.0])
    assert isinstance(result, np.ndarray)
    assert result.shape == (5, 2)


def test_sample_bivariate_invalid_dist():
    """Unsupported distributions should raise ChoiceError."""
    with pytest.raises(ChoiceError):
        sample_bivariate_distribution(10, "invalid", 0.0, [1, 1])


def test_sample_bivariate_exponential_param_length_error():
    """Exponential distribution with wrong param length should raise LengthError."""
    with pytest.raises(LengthError):
        sample_bivariate_distribution(5, "exponential", 0.0, [1.0])


def test_sample_bivariate_weibull_param_length_error():
    """Weibull distribution with wrong param length should raise LengthError."""
    with pytest.raises(LengthError):
        sample_bivariate_distribution(5, "weibull", 0.0, [1.0, 1.0])
