import pandas as pd
import pytest

from gen_surv import generate
from gen_surv.export import export_dataset
from gen_surv.integration import to_sksurv


def test_generate_export_roundtrip(tmp_path):
    """Integration test for generate and export_dataset."""
    df = generate(
        model="cphm",
        n=10,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
        seed=42,
    )
    out = tmp_path / "data.json"
    export_dataset(df, out)
    loaded = pd.read_json(out, orient="table")
    pd.testing.assert_frame_equal(df, loaded)


def test_generate_export_to_sksurv_roundtrip(tmp_path):
    """Full pipeline from generation to scikit-survival array."""
    pytest.importorskip("sksurv.util")
    df = generate(
        model="cphm",
        n=8,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=1.0,
        seed=0,
    )
    out = tmp_path / "data.json"
    export_dataset(df, out)
    loaded = pd.read_json(out, orient="table")
    arr = to_sksurv(loaded)
    assert arr.shape[0] == 8
