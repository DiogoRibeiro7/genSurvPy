import pandas as pd
import pytest

from gen_surv.sklearn_adapter import GenSurvDataGenerator
from gen_surv.validation import ChoiceError


def test_sklearn_generator_dataframe():
    gen = GenSurvDataGenerator(
        "cphm",
        n=4,
        beta=0.2,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=1.0,
    )
    df = gen.fit_transform()
    assert len(df) == 4
    assert {"time", "status"}.issubset(df.columns)


def test_sklearn_generator_dict():
    gen = GenSurvDataGenerator(
        "cphm",
        return_type="dict",
        n=3,
        beta=0.5,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=1.0,
    )
    data = gen.transform()
    assert isinstance(data, dict)
    assert set(data.keys()) >= {"time", "status"}
    assert len(data["time"]) == 3


def test_sklearn_generator_invalid_return_type():
    with pytest.raises(ChoiceError):
        GenSurvDataGenerator(
            "cphm",
            return_type="bad",
            n=1,
            beta=0.5,
            covariate_range=1.0,
            model_cens="uniform",
            cens_par=1.0,
        )


def _estimator():
    return GenSurvDataGenerator(
        "cphm",
        n=5,
        beta=0.5,
        covariate_range=1.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=1,
    )


def test_get_params_exposes_the_model_arguments():
    """scikit-learn cannot introspect ``**kwargs``, so the wrapper must report them."""
    params = _estimator().get_params()

    assert params["model"] == "cphm"
    assert params["return_type"] == "df"
    assert params["n"] == 5
    assert params["beta"] == 0.5
    assert params["seed"] == 1


def test_set_params_accepts_model_arguments():
    estimator = _estimator().set_params(n=17)

    assert estimator.get_params()["n"] == 17
    assert len(estimator.fit_transform()) == 17


def test_set_params_validates_return_type():
    with pytest.raises(ChoiceError):
        _estimator().set_params(return_type="bad")


def test_clone_preserves_the_model_arguments():
    """``clone`` is what pipelines and GridSearchCV use internally.

    Before the wrapper reported its forwarded arguments, cloning silently
    dropped them and the copy raised ``gen_cphm() missing 5 required positional
    arguments`` on first use.
    """
    sklearn_base = pytest.importorskip("sklearn.base")

    original = _estimator()
    copy = sklearn_base.clone(original)

    assert copy.get_params() == original.get_params()
    pd.testing.assert_frame_equal(copy.fit_transform(), original.fit_transform())


def test_clone_after_set_params_carries_the_change():
    sklearn_base = pytest.importorskip("sklearn.base")

    copy = sklearn_base.clone(_estimator().set_params(n=23))

    assert len(copy.fit_transform()) == 23
