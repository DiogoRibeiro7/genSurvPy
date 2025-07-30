import pandas as pd
import pytest
from gen_surv.summary import (
    summarize_survival_dataset,
    check_survival_data_quality,
    _print_summary,
)


def test_summarize_survival_dataset_errors():
    df = pd.DataFrame({"time": [1, 2], "status": [1, 0]})
    # Missing time column
    with pytest.raises(ValueError):
        summarize_survival_dataset(df.drop(columns=["time"]))
    # Missing ID column when specified
    with pytest.raises(ValueError):
        summarize_survival_dataset(df, id_col="id")
    # Missing covariate columns
    with pytest.raises(ValueError):
        summarize_survival_dataset(df, covariate_cols=["bad"])


def test_summarize_survival_dataset_verbose_output(capsys):
    df = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0],
            "status": [1, 0, 1],
            "id": [1, 2, 3],
            "age": [30, 40, 50],
            "group": ["A", "B", "A"],
        }
    )
    summary = summarize_survival_dataset(
        df, id_col="id", covariate_cols=["age", "group"]
    )
    _print_summary(summary, "time", "status", "id", ["age", "group"])
    captured = capsys.readouterr().out
    assert "SURVIVAL DATASET SUMMARY" in captured
    assert "age:" in captured
    assert "Categorical" in captured


def test_check_survival_data_quality_duplicates_and_fix():
    df = pd.DataFrame(
        {
            "time": [1.0, -1.0, 2.0, 1.0],
            "status": [1, 1, 0, 1],
            "id": [1, 1, 2, 1],
        }
    )
    checked, issues = check_survival_data_quality(df, id_col="id", fix_issues=False)
    assert issues["duplicates"]["duplicate_rows"] == 1
    assert issues["duplicates"]["duplicate_ids"] == 2
    fixed, issues_fixed = check_survival_data_quality(
        df, id_col="id", max_time=2.0, fix_issues=True
    )
    assert len(fixed) < len(df)
    assert issues_fixed["modifications"]["rows_dropped"] > 0
