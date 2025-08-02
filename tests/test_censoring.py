import numpy as np

from gen_surv.censoring import (
    WeibullCensoring,
    LogNormalCensoring,
    GammaCensoring,
    rexpocens,
    runifcens,
    rweibcens,
    rlognormcens,
    rgammacens,
)


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


def test_rweibcens_nonnegative():
    times = rweibcens(5, 1.0, 1.5)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times >= 0)


def test_rlognormcens_positive():
    times = rlognormcens(5, 0.0, 1.0)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times > 0)


def test_rgammacens_positive():
    times = rgammacens(5, 2.0, 1.0)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times > 0)


def test_weibull_censoring_class():
    model = WeibullCensoring(scale=1.0, shape=1.5)
    times = model(5)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times >= 0)


def test_lognormal_censoring_class():
    model = LogNormalCensoring(mean=0.0, sigma=1.0)
    times = model(5)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times > 0)


def test_gamma_censoring_class():
    model = GammaCensoring(shape=2.0, scale=1.0)
    times = model(5)
    assert isinstance(times, np.ndarray)
    assert len(times) == 5
    assert np.all(times > 0)
