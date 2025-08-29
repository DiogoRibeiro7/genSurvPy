from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from gen_surv import generate

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
}


@pytest.mark.parametrize("model_key", sorted(MODEL_CONFIGS.keys()))
def test_generate_matches_baseline(
    model_key: str,
    request: pytest.FixtureRequest,
    load_baseline,
    save_baseline,
) -> None:
    cfg = MODEL_CONFIGS[model_key]
    df: pd.DataFrame = generate(**cfg)
    assert "time" in df.columns and (
        "event" in df.columns or "status" in df.columns
    ), "Missing core survival columns."
    baseline_name = f"gen_{model_key}"
    if request.config.getoption("--update-baselines"):
        save_baseline(df, baseline_name)
        pytest.skip(
            f"Baseline {baseline_name} updated; re-run without --update-baselines."
        )
    expected = load_baseline(baseline_name)
    from conftest import assert_frame_numeric_equal

    assert_frame_numeric_equal(df[expected.columns], expected)
