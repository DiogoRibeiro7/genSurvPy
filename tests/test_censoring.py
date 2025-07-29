import numpy as np

from gen_surv.censoring import rexpocens, runifcens


def test_runifcens_range():
    times = runifcens(5, 2.0)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times >= 0)
    assert np.all(times <= 2.0)


def test_rexpocens_nonnegative():
    times = rexpocens(5, 2.0)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times >= 0)
