import numpy as np
import pandas as pd
import pytest

from gen_surv.cmm import gen_cmm, generate_event_times


def test_generate_event_times_reproducible():
    rng = np.random.default_rng(0)
    result = generate_event_times(
        z1=1.0,
        beta=[0.1, 0.2, 0.3],
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        rng=rng,
    )
    assert np.isclose(result["t12"], 0.9168237140025525)
    assert np.isclose(result["t13"], 0.2574241891031173)
    assert np.isclose(result["t23"], 0.030993312969869156)


@pytest.mark.parametrize("model_cens", ["uniform", "exponential"])
def test_gen_cmm_is_reproducible(model_cens):
    """The same seed must reproduce the frame exactly."""
    kwargs = {
        "n": 25,
        "model_cens": model_cens,
        "cens_par": 1.0,
        "beta": [0.1, 0.2, 0.3],
        "covariate_range": 2.0,
        "rate": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "seed": 42,
    }
    pd.testing.assert_frame_equal(gen_cmm(**kwargs), gen_cmm(**kwargs))


def test_gen_cmm_emits_the_counting_process_schema():
    """Columns and dtypes form the documented multistate contract."""
    df = gen_cmm(
        n=20,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        seed=42,
    )

    assert list(df.columns) == [
        "id",
        "start",
        "stop",
        "from_state",
        "to_state",
        "status",
        "X0",
    ]
    assert set(df["from_state"].unique()) <= {1, 2}
    assert set(df["to_state"].unique()) <= {2, 3}
    assert set(df["status"].unique()) <= {0, 1}
    assert (df["stop"] > df["start"]).all()
