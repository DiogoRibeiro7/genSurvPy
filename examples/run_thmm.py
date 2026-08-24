"""Illness-death data as a panel of observed states, from the THMM generator.

The three states are 1 healthy, 2 ill and 3 dead, and every transition
intensity is constant in time -- which is what "time-homogeneous" means. The
states are observed, so this is not a hidden Markov model: there are no
emission parameters. See docs/models/thmm.md.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv import generate

# `rate` is one intensity per transition, in the order 1->2, 1->3, 2->3, and
# `beta` one coefficient per transition in the same order.
df = generate(
    model="thmm",
    n=100,
    model_cens="exponential",
    cens_par=3.0,
    beta=[0.1, 0.2, 0.3],
    covariate_range=1.0,
    rate=[0.2, 0.1, 0.3],
    seed=42,
)

print(df.head())
print()
print(f"{df['id'].nunique()} subjects in {len(df)} rows")

# There is no status column: whether a subject died is read off its last state,
# since state 3 is absorbing.
last = df.sort_values("time").groupby("id").last()
print(f"reached state 3: {int((last['state'] == 3).sum())} of {len(last)}")
