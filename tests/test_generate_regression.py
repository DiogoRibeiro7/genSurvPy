"""Frozen-output regression tests for every generator.

Each model is run with a fixed seed and compared against a stored baseline, so
any change to a sampler's draw order or arithmetic shows up as a failing test
rather than as silently different data.

The baselines are committed under ``tests/baselines``. Regenerate them with::

    pytest tests/test_generate_regression.py --update-baselines

and commit the result **only** when the change in output is intended and
explained in the changelog: a seed producing different data is a break in
reproducibility for anyone who pinned it.

The expected column lists double as the schema contract. Two layouts are
canonical: one row per subject, and counting-process intervals for transition
and recurrent data. ``thmm`` is the documented third case, a state panel.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from gen_surv import generate
from tests.conftest import assert_frame_numeric_equal

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cphm": dict(
        model="cphm",
        n=256,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=0.7,
        seed=1234,
    ),
    "aft_ln": dict(
        model="aft_ln",
        n=256,
        beta=[0.5],
        sigma=0.8,
        model_cens="uniform",
        cens_par=0.8,
        seed=1234,
    ),
    "aft_log_logistic": dict(
        model="aft_log_logistic",
        n=256,
        beta=[0.5],
        shape=1.3,
        scale=1.7,
        model_cens="uniform",
        cens_par=0.8,
        seed=1234,
    ),
    "aft_weibull": dict(
        model="aft_weibull",
        n=256,
        beta=[0.5],
        shape=1.4,
        scale=1.1,
        model_cens="uniform",
        cens_par=0.8,
        seed=1234,
    ),
    "piecewise_exponential": dict(
        model="piecewise_exponential",
        n=256,
        breakpoints=[1.0, 3.0],
        hazard_rates=[0.5, 2.0, 0.2],
        betas=[0.4, -0.2],
        model_cens="uniform",
        cens_par=5.0,
        seed=1234,
    ),
    "competing_risks": dict(
        model="competing_risks",
        n=256,
        n_risks=2,
        baseline_hazards=[0.4, 0.2],
        betas=[[0.8, 0.0], [0.0, -0.5]],
        model_cens="uniform",
        cens_par=5.0,
        max_time=10.0,
        seed=1234,
    ),
    "competing_risks_weibull": dict(
        model="competing_risks_weibull",
        n=256,
        n_risks=2,
        shape_params=[1.2, 0.8],
        scale_params=[2.0, 1.5],
        betas=[[0.6, 0.0], [0.0, -0.4]],
        model_cens="uniform",
        cens_par=5.0,
        max_time=10.0,
        seed=1234,
    ),
    "mixture_cure": dict(
        model="mixture_cure",
        n=256,
        cure_fraction=0.3,
        baseline_hazard=0.8,
        betas_survival=[0.5, -0.2],
        betas_cure=[0.3, 0.1],
        model_cens="uniform",
        cens_par=5.0,
        max_time=10.0,
        seed=1234,
    ),
    "cmm": dict(
        model="cmm",
        n=128,
        model_cens="exponential",
        cens_par=2.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0],
        seed=1234,
    ),
    "thmm": dict(
        model="thmm",
        n=128,
        model_cens="exponential",
        cens_par=2.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.2, 0.3, 0.4],
        seed=1234,
    ),
    "tdcm": dict(
        model="tdcm",
        n=128,
        dist="weibull",
        corr=0.5,
        dist_par=[1.0, 2.0, 1.0, 2.0],
        model_cens="uniform",
        cens_par=5.0,
        beta=[0.5, 0.3],
        lam=1.0,
        seed=1234,
    ),
    "recurrent_events": dict(
        model="recurrent_events",
        n=128,
        process="pwp_gt",
        baseline="weibull",
        baseline_params={"shape": 1.3, "scale": 2.0},
        betas=[0.4, -0.2],
        stratum_effects=[1.0, 1.5],
        followup_time=5.0,
        model_cens="uniform",
        cens_par=8.0,
        seed=1234,
    ),
}

# The layout each generator promises. A change here is a change to the public
# contract, not an implementation detail.
EXPECTED_COLUMNS: Dict[str, List[str]] = {
    "cphm": ["time", "status", "X0"],
    "aft_ln": ["id", "time", "status", "X0"],
    "aft_log_logistic": ["id", "time", "status", "X0"],
    "aft_weibull": ["id", "time", "status", "X0"],
    "piecewise_exponential": ["id", "time", "status", "X0", "X1"],
    "competing_risks": ["id", "time", "status", "X0", "X1"],
    "competing_risks_weibull": ["id", "time", "status", "X0", "X1"],
    "mixture_cure": ["id", "time", "status", "cured", "X0", "X1"],
    "cmm": ["id", "start", "stop", "from_state", "to_state", "status", "X0"],
    "thmm": ["id", "time", "state", "X0"],
    "tdcm": ["id", "start", "stop", "status", "covariate", "tdcov"],
    "recurrent_events": ["id", "start", "stop", "status", "enum", "X0", "X1"],
}


def test_every_registered_model_has_a_regression_case() -> None:
    """A new generator must arrive with a frozen baseline.

    Without this, adding a model to the dispatcher and forgetting to register it
    here would leave it unprotected while the suite still passed.
    """
    from gen_surv.interface import _model_map

    missing = sorted(set(_model_map) - set(MODEL_CONFIGS))
    assert not missing, (
        f"models {missing} are registered in generate() but have no regression "
        "case; add one to MODEL_CONFIGS and commit its baseline"
    )
    assert set(MODEL_CONFIGS) == set(EXPECTED_COLUMNS)


@pytest.mark.parametrize("model_key", sorted(MODEL_CONFIGS.keys()))
def test_generate_matches_baseline(
    model_key: str,
    request: pytest.FixtureRequest,
    load_baseline,
    save_baseline,
) -> None:
    cfg = MODEL_CONFIGS[model_key]
    df: pd.DataFrame = generate(**cfg)

    assert list(df.columns) == EXPECTED_COLUMNS[model_key], (
        f"{model_key} changed its columns; this is a change to the output "
        "contract, not an implementation detail"
    )
    assert len(df) > 0

    baseline_name = f"gen_{model_key}"
    if request.config.getoption("--update-baselines"):
        save_baseline(df, baseline_name)
        pytest.skip(
            f"Baseline {baseline_name} updated; re-run without --update-baselines."
        )

    expected = load_baseline(baseline_name)
    assert_frame_numeric_equal(df[expected.columns], expected)


@pytest.mark.parametrize("model_key", sorted(MODEL_CONFIGS.keys()))
def test_generate_is_deterministic_within_a_run(model_key: str) -> None:
    """The same configuration twice in one process must give the same frame.

    Cheap, and independent of the stored baselines: it catches a generator that
    has picked up a dependency on global state.
    """
    cfg = MODEL_CONFIGS[model_key]

    pd.testing.assert_frame_equal(generate(**cfg), generate(**cfg))
