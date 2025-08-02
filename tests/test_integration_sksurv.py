import pandas as pd
import pytest

from gen_surv.integration import to_sksurv


def test_to_sksurv():
    # Optional integration test; skipped when scikit-survival is not installed.
    pytest.importorskip("sksurv.util")
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    arr = to_sksurv(df)
    assert arr.dtype.names == ("status", "time")
    assert arr.shape[0] == 2
