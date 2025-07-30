import os

import pandas as pd
import pytest

from gen_surv import export_dataset, generate


def test_export_dataset_csv(tmp_path):
    df = generate(
        model="cphm",
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
    )
    out_file = tmp_path / "data.csv"
    export_dataset(df, str(out_file))
    assert out_file.exists()
    loaded = pd.read_csv(out_file)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded)


def test_export_dataset_json(tmp_path):
    df = generate(
        model="cphm",
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
    )
    out_file = tmp_path / "data.json"
    export_dataset(df, str(out_file))
    assert out_file.exists()
    loaded = pd.read_json(out_file, orient="table")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded)


def test_export_dataset_feather_and_invalid(tmp_path):
    df = generate(
        model="cphm",
        n=5,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
    )
    feather_file = tmp_path / "data.feather"
    export_dataset(df, str(feather_file))
    assert feather_file.exists()
    loaded = pd.read_feather(feather_file)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded)

    with pytest.raises(ValueError):
        export_dataset(df, str(tmp_path / "data.txt"), fmt="txt")
