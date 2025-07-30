import pandas as pd

from gen_surv.integration import to_sksurv


def test_to_sksurv():
    df = pd.DataFrame({"time": [1.0, 2.0], "status": [1, 0]})
    arr = to_sksurv(df)
    assert arr.dtype.names == ("status", "time")
    assert arr.shape[0] == 2
