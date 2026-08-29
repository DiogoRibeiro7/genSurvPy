"""Compute the preregistered H1--H4 estimands from processed results.

    python scripts/analyze_hypotheses.py --processed results/processed

Writes ``hypotheses.parquet`` and ``hypotheses.json``. The manuscript and
tables should consume those artifacts rather than recomputing hypothesis
quantities ad hoc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.hypotheses import analyse_hypotheses  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", default=str(HERE.parent / "results" / "processed")
    )
    arguments = parser.parse_args()

    processed = Path(arguments.processed)
    summary_path = processed / "summary.parquet"
    if not summary_path.exists():
        print(f"no summary at {summary_path}; run aggregate_results.py first")
        return 1

    hypotheses = analyse_hypotheses(pd.read_parquet(summary_path))
    parquet_path = processed / "hypotheses.parquet"
    json_path = processed / "hypotheses.json"
    hypotheses.to_parquet(parquet_path, index=False)
    json_path.write_text(
        json.dumps(hypotheses.to_dict("records"), indent=2), encoding="utf-8"
    )
    print(f"hypotheses     {len(hypotheses)} rows -> {parquet_path}")
    print(f"json           {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
