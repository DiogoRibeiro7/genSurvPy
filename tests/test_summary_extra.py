import pandas as pd
import pytest
from gen_surv.summary import check_survival_data_quality, compare_survival_datasets
from gen_surv import generate


def test_check_survival_data_quality_fix_issues():
    df = pd.DataFrame(
        {
            "time": [1.0, -0.5, None, 1.0],
            "status": [1, 2, 0, 1],
            "id": [1, 2, 3, 1],
        }
    )
    fixed, issues = check_survival_data_quality(
        df,
        id_col="id",
        max_time=2.0,
        fix_issues=True,
    )
    assert issues["modifications"]["rows_dropped"] == 2
    assert issues["modifications"]["values_fixed"] == 1
    assert len(fixed) == 2


def test_check_survival_data_quality_no_fix():
    """Issues should be reported but data left unchanged when fix_issues=False."""
    df = pd.DataFrame({"time": [-1.0, 2.0], "status": [3, 1]})
    checked, issues = check_survival_data_quality(df, max_time=1.0, fix_issues=False)
    # Data is returned unmodified
    pd.testing.assert_frame_equal(df, checked)
    assert issues["invalid_values"]["negative_time"] == 1
    assert issues["invalid_values"]["excessive_time"] == 1
    assert issues["invalid_values"]["invalid_status"] == 1


def test_compare_survival_datasets_basic():
    ds1 = generate(model="cphm", n=5, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=1.0)
    ds2 = generate(model="cphm", n=5, model_cens="uniform", cens_par=1.0, beta=1.0, covariate_range=1.0)
    comparison = compare_survival_datasets({"A": ds1, "B": ds2})
    assert set(["A", "B"]).issubset(comparison.columns)
    assert "n_subjects" in comparison.index


def test_compare_survival_datasets_with_covariates_and_empty_error():
    ds = generate(model="cphm", n=3, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=1.0)
    comparison = compare_survival_datasets({"only": ds}, covariate_cols=["X0"])
    assert "only" in comparison.columns
    assert "X0_mean" in comparison.index
    with pytest.raises(ValueError):
        compare_survival_datasets({})
