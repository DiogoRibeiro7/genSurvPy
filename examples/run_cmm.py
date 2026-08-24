"""Illness-death data in counting-process form, from the CMM generator.

Each subject contributes one row per transition it was at risk of, so the frame
is longer than ``n``. See docs/models/cmm.md for what the columns mean.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv import generate

# `rate` is three (intensity, shape) pairs, one per transition: 1->2, 1->3 and
# 2->3. `beta` is one coefficient per transition, in the same order.
df = generate(
    model="cmm",
    n=100,
    model_cens="exponential",
    cens_par=2.0,
    beta=[0.1, 0.2, 0.3],
    covariate_range=1.0,
    rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0],
    seed=42,
)

print(df.head())
print()
print(f"{df['id'].nunique()} subjects in {len(df)} rows")
print(df.groupby(["from_state", "to_state"])["status"].agg(["size", "sum"]))
