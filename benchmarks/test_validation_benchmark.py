import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from gen_surv.validation import ensure_positive_sequence


def test_positive_sequence_benchmark(benchmark):
    seq = np.random.rand(10000) + 1.0
    benchmark(ensure_positive_sequence, seq, "seq")
