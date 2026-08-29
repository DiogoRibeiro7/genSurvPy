"""Parity with the R ``genSurv`` package, against frozen reference output.

Four generators were ported from R: ``genCPHM``, ``genCMM``, ``genTHMM`` and
``genTDCM``. Divergence from them should be detected rather than argued about,
so ``scripts/generate_r_fixtures.R`` freezes real R output into
``tests/fixtures/r_parity/`` and these tests compare against it. CI never needs
R installed.

**Parity is on distributions, not values.** R draws from the Mersenne Twister
and we draw from PCG64, so identical numbers are impossible however faithful
the port. Each test therefore compares a statistic -- a rate, a proportion, an
occurrence/exposure intensity -- and allows for the Monte Carlo error on both
sides.

Both sides are fixed: the R output is frozen and the Python seeds are listed
here, so these tests are deterministic. The four-sigma bands are wide enough to
absorb the sampling error that remains and narrow enough that a real divergence
fails. A single sample would not do: three separate comparisons during this
work looked significant at one seed and vanished over eight.

``tdcm`` is the exception, and deliberately so -- see
``test_tdcm_diverges_because_the_r_bivariate_sampler_is_defective``.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from gen_surv import gen_cmm, gen_cphm, gen_tdcm, gen_thmm
from gen_surv.bivariate import sample_bivariate_distribution

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "r_parity"

#: Fixed so the comparison is deterministic. Several seeds rather than one
#: because a single sample gives no estimate of its own error.
SEEDS = (101, 102, 103, 104, 105)

#: These mirror `scripts/generate_r_fixtures.R` and must be kept in step with
#: it. `test_fixtures_have_the_expected_size` fails if they drift.
CPHM = dict(n=20000, beta=0.5, covariate_range=2.0, model_cens="uniform", cens_par=1.0)
CMM = dict(
    n=25000,
    model_cens="uniform",
    cens_par=1.0,
    beta=[0.1, 0.2, 0.3],
    covariate_range=1.0,
    rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0],
)
THMM = dict(
    n=25000,
    model_cens="uniform",
    cens_par=1.0,
    beta=[0.1, 0.2, 0.3],
    covariate_range=1.0,
    rate=[0.2, 0.3, 0.4],
)
TDCM = dict(
    n=20000,
    dist="weibull",
    corr=0.5,
    dist_par=[1.0, 2.0, 1.0, 2.0],
    model_cens="uniform",
    cens_par=1.0,
    beta=[0.5, 0.3],
    lam=1.0,
)

#: R's `genCMM` writes both competing rows for every subject and labels them in
#: its own order: `aux1` carries `trans = 1` and takes `event = 1` when `t13`
#: wins, so **trans 1 is the 1 -> 3 edge and trans 2 is the 1 -> 2 edge**. The
#: parameters map straight across -- R's `t12` uses `rate[1:2]` where ours uses
#: `rate[0:2]` -- only the labels are ordered differently.
R_TRANS_TO_EDGE = {1: (1, 3), 2: (1, 2), 3: (2, 3)}

#: Three, not four. Both sides are frozen -- the R output on disk, the Python
#: seeds above -- so this is not guarding against repeated sampling; it is the
#: width of the band a real divergence has to clear. Measured discriminating
#: power at this width, by perturbing the Python side and re-running:
#:
#:   cphm censoring rate    a 10% change in `beta` is caught
#:   cmm intensity 1->2     a 15% change in `rate[0]` is caught, 10% is not
#:   thmm state occupancy   a 15% change in `rate[0]` is caught, 10% is not
#:
#: Tightening further needs a larger frozen fixture, not a smaller SIGMA.
SIGMA = 3.0


def _load(name: str) -> pd.DataFrame:
    path = FIXTURES / f"{name}.csv.gz"
    if not path.exists():  # pragma: no cover - the fixtures are committed
        raise AssertionError(
            f"Missing R fixture {path}. Regenerate with "
            "`Rscript scripts/generate_r_fixtures.R` (needs R and genSurv)."
        )
    return pd.read_csv(path)


def _agrees(
    label: str, r_value: float, r_se: float, py_values: Sequence[float]
) -> None:
    """Fail if R and Python differ by more than ``SIGMA`` combined standard errors."""
    py = np.asarray(py_values, dtype=float)
    combined = float(np.sqrt(r_se**2 + py.var(ddof=1) / py.size))
    z = (py.mean() - r_value) / combined

    assert abs(z) < SIGMA, (
        f"{label}: R={r_value:.5f}, python={py.mean():.5f} "
        f"(over {py.size} seeds), difference {py.mean() - r_value:+.5f} "
        f"= {z:+.2f} combined standard errors. Either the port has drifted "
        f"from R or this tolerance is wrong; do not widen it without deciding "
        f"which."
    )


def _proportion_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1.0 - p), 1e-12) / n))


# --------------------------------------------------------------------------
# The fixtures themselves
# --------------------------------------------------------------------------


def test_fixtures_record_their_provenance() -> None:
    """A frozen reference is only interpretable if it says what produced it."""
    version = (FIXTURES / "VERSION.txt").read_text(encoding="utf-8")
    assert "genSurv" in version
    assert "R " in version


@pytest.mark.parametrize(
    "name,subject_column,expected",
    [
        ("cphm", None, CPHM["n"]),
        ("cmm", "id", CMM["n"]),
        ("thmm", "PTNUM", THMM["n"]),
        ("tdcm", "id", TDCM["n"]),
    ],
)
def test_fixtures_have_the_expected_size(
    name: str, subject_column: str | None, expected: int
) -> None:
    """Guards the coupling to `scripts/generate_r_fixtures.R`.

    The parameters live in two places -- that script and this module -- and
    nothing else would notice if one were regenerated at a different `n`.
    """
    frame = _load(name)
    actual = len(frame) if subject_column is None else frame[subject_column].nunique()
    assert actual == expected, (
        f"{name} fixture holds {actual} subjects, not {expected}. If "
        "generate_r_fixtures.R changed, update the parameters here to match."
    )


# --------------------------------------------------------------------------
# CPHM
# --------------------------------------------------------------------------


def test_cphm_censoring_rate_matches_r() -> None:
    r = _load("cphm")
    r_rate = float((r["status"] == 0).mean())

    py = [
        float((gen_cphm(**CPHM, seed=s)["status"] == 0).mean())  # type: ignore[arg-type]
        for s in SEEDS
    ]
    _agrees("cphm censoring rate", r_rate, _proportion_se(r_rate, len(r)), py)


def test_cphm_mean_time_matches_r() -> None:
    r = _load("cphm")
    r_mean = float(r["time"].mean())
    r_se = float(r["time"].std(ddof=1) / np.sqrt(len(r)))

    py = [float(gen_cphm(**CPHM, seed=s)["time"].mean()) for s in SEEDS]  # type: ignore[arg-type]
    _agrees("cphm mean time", r_mean, r_se, py)


def test_cphm_time_distribution_matches_r() -> None:
    """The whole distribution, not two moments of it."""
    r = _load("cphm")
    frame = gen_cphm(**CPHM, seed=SEEDS[0])  # type: ignore[arg-type]

    result = stats.ks_2samp(r["time"].to_numpy(), frame["time"].to_numpy())
    assert result.pvalue > 0.001, (
        f"cphm event times diverge from R: KS D={result.statistic:.5f}, "
        f"p={result.pvalue:.3g}"
    )


# --------------------------------------------------------------------------
# CMM
# --------------------------------------------------------------------------


@pytest.mark.parametrize("r_trans", sorted(R_TRANS_TO_EDGE))
def test_cmm_transition_intensities_match_r(r_trans: int) -> None:
    """Events divided by time at risk, per edge.

    This is the quantity `cmm` is *about*, and it is invariant to how either
    implementation lays its rows out, which is what makes it the right thing to
    compare across two packages with different schemas.
    """
    origin, destination = R_TRANS_TO_EDGE[r_trans]

    r = _load("cmm")
    rows = r[r["trans"] == r_trans]
    r_events = int(rows["event"].sum())
    r_exposure = float((rows["stop"] - rows["start"]).sum())
    r_rate = r_events / r_exposure
    # An occurrence/exposure rate has relative standard error 1/sqrt(events).
    r_se = r_rate / np.sqrt(max(r_events, 1))

    py = []
    for seed in SEEDS:
        frame = gen_cmm(**CMM, seed=seed)  # type: ignore[arg-type]
        edge = frame[
            (frame["from_state"] == origin) & (frame["to_state"] == destination)
        ]
        exposure = float((edge["stop"] - edge["start"]).sum())
        py.append(float(edge["status"].sum()) / exposure)

    _agrees(f"cmm intensity {origin}->{destination}", r_rate, r_se, py)


# --------------------------------------------------------------------------
# THMM
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", [1, 2, 3])
def test_thmm_final_state_occupancy_matches_r(state: int) -> None:
    r = _load("thmm")
    r_last = r.sort_values("time").groupby("PTNUM").last()
    r_share = float((r_last["state"] == state).mean())

    py = []
    for seed in SEEDS:
        frame = gen_thmm(**THMM, seed=seed)  # type: ignore[arg-type]
        last = frame.sort_values("time").groupby("id").last()
        py.append(float((last["state"] == state).mean()))

    _agrees(
        f"thmm share ending in state {state}",
        r_share,
        _proportion_se(r_share, len(r_last)),
        py,
    )


# --------------------------------------------------------------------------
# TDCM: a divergence we intend to keep
# --------------------------------------------------------------------------


def test_tdcm_diverges_because_the_r_bivariate_sampler_is_defective() -> None:
    """`gen_tdcm` must NOT match R here, and this pins down why.

    The time computation is identical -- R's

        t <- -(log(1 - u) + lambda * b1 * exp(beta1 * z1) * (1 - exp(beta2)))
             / (lambda * exp(beta1 * z1 + beta2))

    is our `(log_term + x * (exp(beta[1]) - 1)) / (lam * exp(...))` rearranged.
    The divergence is upstream, in the bivariate draw both feed on.

    Asked for Weibull marginals with `dist.par = c(1, 2, 1, 2)`, R's `dgBIV`
    returns something with mean 2 and median 2*log(2): chi-square with two
    degrees of freedom, which is Exponential with mean 2. It ignores the
    requested parameterisation. Ours inverts the Weibull CDF through a Gaussian
    copula and returns the distribution asked for.

    That is the same class of defect the 2.0.0 release corrected for the
    exponential case -- `chi2(1) / 2` where an exponential was requested -- and
    this test records it for the Weibull case, so nobody "fixes" our sampler
    into agreement with R.
    """
    draw = sample_bivariate_distribution(
        50000, "weibull", 0.5, [1.0, 2.0, 1.0, 2.0], seed=7
    )

    # Ours is the requested Weibull(shape=2, scale=1).
    for column in (0, 1):
        ours = stats.kstest(draw[:, column], "weibull_min", args=(2.0, 0.0, 1.0))
        assert ours.pvalue > 0.001, (
            f"our bivariate margin {column} no longer matches the Weibull it "
            f"was asked for: D={ours.statistic:.4f}, p={ours.pvalue:.3g}"
        )

    # R's is not, and the R fixture carries the consequence: shorter follow-up
    # and more events than ours, because its covariate has a heavier tail.
    r = _load("tdcm")
    per_subject = r.groupby("id").agg(exit=("stop", "max"), event=("event", "max"))
    frame = gen_tdcm(**TDCM, seed=SEEDS[0])  # type: ignore[arg-type]

    assert per_subject["exit"].mean() < frame["stop"].mean(), (
        "R's tdcm follow-up is no longer shorter than ours. The divergence "
        "documented here has changed; re-check R's dgBIV before editing this."
    )
    assert per_subject["event"].mean() > frame["status"].mean()


def test_tdcm_still_agrees_with_r_on_the_crossover_rule() -> None:
    """The divergence is confined to the bivariate draw.

    Both packages mark the time-dependent covariate the same way -- it is 1
    exactly when the crossover is observed before exit -- so the *rule* can be
    compared even though the draws feeding it cannot. R expresses it by
    splitting the row at the crossover; we record the value at exit.
    """
    r = _load("tdcm")
    split = r.groupby("id").size()
    crossed_and_observed = r[r["tdcov"] == 1]["id"].nunique()

    assert set(split.unique()) <= {1, 2}, "R emits at most two rows per subject"
    assert crossed_and_observed == int((split == 2).sum()), (
        "In R every second row is the post-crossover interval and carries "
        "tdcov=1; that correspondence no longer holds in the fixture."
    )

    frame = gen_tdcm(**TDCM, seed=SEEDS[0])  # type: ignore[arg-type]
    assert len(frame) == TDCM["n"], "ours stays one row per subject"
    assert set(frame["tdcov"].unique()) <= {0.0, 1.0}
