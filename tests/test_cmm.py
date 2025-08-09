import numpy as np
import pandas as pd

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


def test_gen_cmm_uniform_reproducible():
    df = gen_cmm(
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        seed=42,
    )
    expected = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "start": [0.0] * 5,
            "stop": [
                0.18915094163423693,
                0.6785349983450479,
                0.046776460564183294,
                0.12811363267554587,
                0.45038631001973155,
            ],
            "status": [1, 1, 1, 0, 0],
            "X0": [
                1.5479119272347037,
                0.8777564989945617,
                1.7171958398225217,
                1.3947360581187287,
                0.1883555828087116,
            ],
            "transition": [2.0, 2.0, 2.0, float("nan"), float("nan")],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_gen_cmm_exponential_reproducible():
    df = gen_cmm(
        n=5,
        model_cens="exponential",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        seed=42,
    )
    expected = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "start": [0.0] * 5,
            "stop": [
                0.18915094163423693,
                0.6785349983450479,
                0.046776460564183294,
                0.07929383504134148,
                0.5750008479681584,
            ],
            "status": [1, 1, 1, 0, 1],
            "X0": [
                1.5479119272347037,
                0.8777564989945617,
                1.7171958398225217,
                1.3947360581187287,
                0.1883555828087116,
            ],
            "transition": [2.0, 2.0, 2.0, float("nan"), 1.0],
        }
    )
    pd.testing.assert_frame_equal(df, expected)
