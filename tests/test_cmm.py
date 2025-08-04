import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gen_surv.cmm import gen_cmm, generate_event_times


def test_generate_event_times_reproducible():
    np.random.seed(0)
    result = generate_event_times(
        z1=1.0,
        beta=[0.1, 0.2, 0.3],
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    assert np.isclose(result["t12"], 0.7201370350469476)
    assert np.isclose(result["t13"], 1.0282691393768246)
    assert np.isclose(result["t23"], 0.6839405281667484)


def test_gen_cmm_uniform_reproducible():
    np.random.seed(42)
    df = gen_cmm(
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    expected = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "start": [0.0] * 5,
            "stop": [
                0.019298197410170713,
                0.05808361216819946,
                0.5550989864862181,
                0.2117537394012932,
                0.19451374567187332,
            ],
            "status": [1, 0, 1, 1, 1],
            "X0": [
                0.749080237694725,
                1.9014286128198323,
                1.4639878836228102,
                1.1973169683940732,
                0.31203728088487304,
            ],
            "transition": [1.0, float("nan"), 2.0, 1.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_gen_cmm_exponential_reproducible():
    np.random.seed(42)
    df = gen_cmm(
        n=5,
        model_cens="exponential",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    expected = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "start": [0.0] * 5,
            "stop": [
                0.019298197410170713,
                0.059838768608680676,
                0.5550989864862181,
                0.2117537394012932,
                0.19451374567187332,
            ],
            "status": [1, 0, 1, 1, 1],
            "X0": [
                0.749080237694725,
                1.9014286128198323,
                1.4639878836228102,
                1.1973169683940732,
                0.31203728088487304,
            ],
            "transition": [1.0, float("nan"), 2.0, 1.0, 1.0],
        }
    )
    pd.testing.assert_frame_equal(df, expected)
