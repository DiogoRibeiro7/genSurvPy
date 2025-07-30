import pandas as pd
import pyreadr

from gen_surv.export import export_dataset


def test_export_dataset_rds(tmp_path):
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    out = tmp_path / "data.rds"
    export_dataset(df, out)
    assert out.exists()
    result = pyreadr.read_r(out)[None]
    result = result.astype(df.dtypes.to_dict())
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df)
