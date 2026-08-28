"""Evaluation measures, kept in four groups the paper must not conflate.

1. **Recovery of the truth.** Integrated squared and absolute error between
   the predicted and the true conditional survival function. Only possible
   because the mechanism is known, and the primary outcome of this study.
2. **Discrimination.** Harrell's and Uno's concordance, and time-dependent
   AUC. These measure *ranking*. A model can rank every subject correctly
   while assigning probabilities that are all far too high.
3. **Prediction error and calibration.** Brier score, integrated Brier score,
   and a grouped calibration error. These use observed, censored outcomes, so
   they are what an analyst without the truth can actually compute -- which
   makes "does this detect what MISE detects?" the operational question of the
   paper.
4. **Parameter recovery**, handled in :mod:`experiments` because it depends on
   which DGP parameters a given estimator even has an analogue for.

Every integral is a trapezoid rule on the prespecified grid. The horizon
``tau`` is fixed in the protocol before any results are seen; choosing it
afterwards would let the headline result be tuned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# scikit-survival 0.25.0 still calls `np.trapz`, which NumPy 2.0 removed in
# favour of `np.trapezoid`. Without this every Brier and AUC call returns
# `AttributeError: module 'numpy' has no attribute 'trapz'`, which our error
# capture turns into a silent NaN column -- a whole family of metrics quietly
# missing from the study. The two functions are the same implementation under
# different names, so the alias is exact rather than an approximation. Remove
# once scikit-survival ships a NumPy 2 fix.
if not hasattr(np, "trapz"):  # pragma: no cover - environment dependent
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

__all__ = [
    "expected_mortality",
    "d_calibration",
    "antolini_concordance",
    "integrated_squared_error",
    "integrated_absolute_error",
    "truth_recovery",
    "discrimination",
    "prediction_error",
    "grouped_calibration_error",
    "evaluate_all",
]


def _restrict(
    times: NDArray[np.float64], tau: float
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    grid = np.asarray(times, dtype=float)
    keep = grid <= tau
    if keep.sum() < 2:
        raise ValueError(
            f"the evaluation grid has fewer than two points at or below tau={tau}; "
            "the integrated losses are undefined"
        )
    return grid[keep], keep


def integrated_squared_error(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    times: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    r"""Per-subject :math:`\int_0^\tau [\hat S_i(t) - S_i(t)]^2 dt`.

    Returns one value per subject; the study reports the mean (MISE) and its
    Monte Carlo error. Kept per-subject because the distribution across
    subjects is itself interesting: misspecification often hurts the tails of
    the covariate distribution far more than the centre.
    """
    grid, keep = _restrict(times, tau)
    residual = (predicted[:, keep] - truth[:, keep]) ** 2
    return np.trapezoid(residual, grid, axis=1)


def integrated_absolute_error(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    times: NDArray[np.float64],
    tau: float,
) -> NDArray[np.float64]:
    r"""Per-subject :math:`\int_0^\tau |\hat S_i(t) - S_i(t)| dt`.

    Reported alongside the squared error because it is on the scale of the
    survival probability itself, so "0.05" means an average absolute error of
    five percentage points sustained over the horizon -- which a reader can
    judge as practically important or not. The squared error cannot be read
    that way.
    """
    grid, keep = _restrict(times, tau)
    residual = np.abs(predicted[:, keep] - truth[:, keep])
    return np.trapezoid(residual, grid, axis=1)


def truth_recovery(
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    times: NDArray[np.float64],
    tau: float,
) -> dict[str, float]:
    """MISE, MIAE and the tail of the per-subject error distribution."""
    ise = integrated_squared_error(predicted, truth, times, tau)
    iae = integrated_absolute_error(predicted, truth, times, tau)

    return {
        "mise": float(np.mean(ise)),
        "mise_sd": float(np.std(ise, ddof=1)) if ise.size > 1 else float("nan"),
        "miae": float(np.mean(iae)),
        "miae_p90": float(np.quantile(iae, 0.90)),
        "miae_max": float(np.max(iae)),
        # Normalised by the horizon so it reads as a mean absolute error in
        # survival probability, comparable across scenarios with different tau.
        "mean_absolute_survival_error": float(np.mean(iae) / tau),
    }


def discrimination(
    risk: NDArray[np.float64],
    train_time: NDArray[np.float64],
    train_event: NDArray[np.bool_],
    test_time: NDArray[np.float64],
    test_event: NDArray[np.bool_],
    tau: float,
) -> dict[str, float]:
    """Concordance measures. These say nothing about absolute probabilities."""
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
    from sksurv.util import Surv

    out: dict[str, float] = {}

    harrell = concordance_index_censored(
        np.asarray(test_event, dtype=bool), np.asarray(test_time, float), risk
    )
    out["c_index_harrell"] = float(harrell[0])

    # Uno's estimator reweights by the censoring distribution and is the one to
    # prefer when censoring is heavy, which is precisely the regime this study
    # stresses. It can fail when the truncation time leaves too few at risk.
    try:
        train = Surv.from_arrays(
            np.asarray(train_event, bool), np.asarray(train_time, float)
        )
        test = Surv.from_arrays(
            np.asarray(test_event, bool), np.asarray(test_time, float)
        )
        uno = concordance_index_ipcw(train, test, risk, tau=tau)
        out["c_index_uno"] = float(uno[0])
    except Exception as exception:  # noqa: BLE001 - recorded, not hidden
        out["c_index_uno"] = float("nan")
        out["c_index_uno_error"] = f"{type(exception).__name__}: {exception}"

    return out


def prediction_error(
    predicted: NDArray[np.float64],
    times: NDArray[np.float64],
    train_time: NDArray[np.float64],
    train_event: NDArray[np.bool_],
    test_time: NDArray[np.float64],
    test_event: NDArray[np.bool_],
    tau: float,
) -> dict[str, float]:
    """Brier score, integrated Brier score, and time-dependent AUC.

    All three are computed from censored observations with inverse-probability-
    of-censoring weights, so they are available to an analyst who does not know
    the truth. That is the point: the paper asks which of these detect the
    error that MISE measures directly.
    """
    from sksurv.metrics import (
        brier_score,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )
    from sksurv.util import Surv

    train = Surv.from_arrays(
        np.asarray(train_event, bool), np.asarray(train_time, float)
    )
    test = Surv.from_arrays(np.asarray(test_event, bool), np.asarray(test_time, float))

    grid = np.asarray(times, dtype=float)
    train_followup = np.asarray(train_time, dtype=float)
    test_followup = np.asarray(test_time, dtype=float)

    # scikit-survival requires evaluation times to sit strictly inside the
    # observed follow-up support. Use the support restriction explicitly and
    # reserve the "at tau" names for calculations that are truly at tau.
    lower = max(float(np.min(train_followup)), float(np.min(test_followup)))
    support_upper = min(float(np.max(train_followup)), float(np.max(test_followup)))
    usable = (grid > lower) & (grid <= float(tau)) & (grid < support_upper)

    out: dict[str, float] = {}
    tau_index = int(np.searchsorted(grid, tau, side="left"))
    tau_on_grid = tau_index < grid.size and np.isclose(grid[tau_index], tau)
    tau_usable = bool(tau_on_grid and tau > lower and tau < support_upper)
    out["brier_at_tau_time"] = float(tau) if tau_usable else float("nan")
    out["auc_at_tau_time"] = float(tau) if tau_usable else float("nan")

    if usable.sum() < 2:
        out.update(
            {
                "brier_at_tau": float("nan"),
                "integrated_brier_score": float("nan"),
                "auc_mean": float("nan"),
                "auc_at_tau": float("nan"),
                "prediction_error_note": (
                    "no grid point inside the usable follow-up range"
                ),
            }
        )
        return out

    sub_grid = grid[usable]
    sub_predicted = predicted[:, usable]

    try:
        _, scores = brier_score(train, test, sub_predicted, sub_grid)
        out["integrated_brier_score"] = float(
            integrated_brier_score(train, test, sub_predicted, sub_grid)
        )
        if tau_usable:
            _, tau_score = brier_score(
                train, test, predicted[:, [tau_index]], np.asarray([tau], dtype=float)
            )
            out["brier_at_tau"] = float(tau_score[0])
        else:
            out["brier_at_tau"] = float("nan")
    except Exception as exception:  # noqa: BLE001
        out["brier_at_tau"] = float("nan")
        out["integrated_brier_score"] = float("nan")
        out["brier_error"] = f"{type(exception).__name__}: {exception}"

    try:
        auc, mean_auc = cumulative_dynamic_auc(
            train, test, 1.0 - sub_predicted, sub_grid
        )
        out["auc_mean"] = float(mean_auc)
        if tau_usable:
            tau_auc, _ = cumulative_dynamic_auc(
                train,
                test,
                1.0 - predicted[:, [tau_index]],
                np.asarray([tau], dtype=float),
            )
            out["auc_at_tau"] = float(tau_auc[0])
        else:
            out["auc_at_tau"] = float("nan")
    except Exception as exception:  # noqa: BLE001
        out["auc_mean"] = float("nan")
        out["auc_at_tau"] = float("nan")
        out["auc_error"] = f"{type(exception).__name__}: {exception}"

    return out


def grouped_calibration_error(
    predicted_at_tau: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.bool_],
    tau: float,
    n_groups: int = 10,
) -> dict[str, float]:
    """Observed-versus-predicted survival at ``tau``, by predicted-risk group.

    Subjects are split into ``n_groups`` equal-sized bins by predicted
    :math:`\\hat S(\\tau)`. Within each bin the observed survival is estimated
    by Kaplan-Meier, which handles the censoring that makes a naive event rate
    biased. The reported error is the mean absolute gap across bins, weighted
    by bin size.

    This is the practically available counterpart to MISE: it needs no
    knowledge of the truth, only the data an analyst would have. Comparing the
    two is how the paper answers whether miscalibration is detectable in
    practice.
    """
    from lifelines import KaplanMeierFitter

    predicted = np.asarray(predicted_at_tau, dtype=float)
    order = np.argsort(predicted)
    bins = np.array_split(order, n_groups)

    gaps: list[float] = []
    weights: list[int] = []
    for indices in bins:
        if indices.size == 0:
            continue
        fitter = KaplanMeierFitter()
        try:
            fitter.fit(
                np.asarray(time, float)[indices], np.asarray(event, bool)[indices]
            )
            observed = float(fitter.predict(tau))
        except Exception:  # noqa: BLE001 - an empty or degenerate bin
            continue
        gaps.append(abs(observed - float(np.mean(predicted[indices]))))
        weights.append(int(indices.size))

    if not gaps:
        return {"calibration_error": float("nan"), "calibration_groups": 0}

    gap_array = np.asarray(gaps, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    return {
        "calibration_error": float(np.average(gap_array, weights=weight_array)),
        "calibration_error_max": float(np.max(gap_array)),
        "calibration_groups": int(gap_array.size),
    }


# ---------------------------------------------------------------------------
# Measures defined on the predicted distribution rather than on a score
# ---------------------------------------------------------------------------


def _survival_at(
    predicted: NDArray[np.float64],
    grid: NDArray[np.float64],
    times: NDArray[np.float64],
    step_like: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Each subject's own predicted survival at its own time.

    Distribution metrics compare the survival probabilities an estimator
    actually reports. The non-parametric estimators return step functions, so
    linear interpolation would create probabilities that were never predicted
    and can break the tie structure Antolini's index is meant to expose. The
    lookup is therefore right-continuous for rows with flat stretches. Smooth
    rows are interpolated between reported values, avoiding a coarse-grid
    discretisation artefact for parametric curves.
    """
    if step_like is None:
        step_like = _step_like_rows(predicted)

    clipped = np.clip(np.asarray(times, dtype=float), grid[0], grid[-1])
    columns = np.clip(np.searchsorted(grid, clipped, side="left"), 0, grid.size - 1)
    out = np.empty(predicted.shape[0], dtype=float)
    for row in range(predicted.shape[0]):
        if step_like[row]:
            out[row] = float(predicted[row, columns[row]])
        else:
            out[row] = float(np.interp(clipped[row], grid, predicted[row]))
    return out


def _step_like_rows(predicted: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Rows with flat stretches should be evaluated as step functions."""
    return np.any(np.isclose(np.diff(predicted, axis=1), 0.0, atol=1e-12), axis=1)


def _survival_at_common_time(
    predicted: NDArray[np.float64],
    grid: NDArray[np.float64],
    time: float,
    step_like: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """All subjects' predicted survival at one time, vectorised."""
    if step_like is None:
        step_like = _step_like_rows(predicted)

    clipped = float(np.clip(time, grid[0], grid[-1]))
    step_column = int(
        np.clip(np.searchsorted(grid, clipped, side="left"), 0, grid.size - 1)
    )
    out = predicted[:, step_column].astype(float, copy=True)

    smooth = ~step_like
    if smooth.any():
        if clipped <= grid[0]:
            out[smooth] = predicted[smooth, 0]
        elif clipped >= grid[-1]:
            out[smooth] = predicted[smooth, -1]
        else:
            right = int(np.searchsorted(grid, clipped, side="right"))
            left = right - 1
            weight = (clipped - grid[left]) / (grid[right] - grid[left])
            out[smooth] = (
                predicted[smooth, left] * (1.0 - weight)
                + predicted[smooth, right] * weight
            )

    return out


def expected_mortality(
    predicted: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""One risk score, computed the same way for every model.

    .. math::

        r_i = \sum_k \hat H(t_k \mid x_i)
            = -\sum_k \log \hat S(t_k \mid x_i)

    the summed cumulative hazard over the evaluation grid — the "expected
    mortality" transformation of Ishwaran et al. (2008).

    **Why this exists.** Sonabend et al. (2022) identify three forms of
    C-hacking, and the third is evaluating distribution predictions with a
    discrimination measure without justifying the transformation used to get
    there. This study previously scored Harrell's concordance on each model's
    *native* score: a partial hazard for Cox, a negative expected survival time
    for the Weibull AFT, and scikit-survival's expected mortality for the forest
    and the boosted model. Three different mathematical objects compared with
    one measure, which is the comparison Sonabend et al. call meaningless.

    Deriving the score from the predicted survival curve in one fixed way makes
    the concordances comparable across estimators, because the only thing that
    differs between them is then the curve itself. The native scores are still
    reported, separately and under their own name — Sonabend et al. note that
    reporting both is legitimate, and only conflating them is not.

    Antolini's index needs no transformation at all and remains the measure to
    prefer where the mechanism is non-proportional.
    """
    grid = np.asarray(grid, dtype=float)
    safe = np.clip(predicted, 1e-12, 1.0)
    cumulative_hazard = -np.log(safe)
    return np.trapezoid(cumulative_hazard, grid, axis=1)


def d_calibration(
    predicted: NDArray[np.float64],
    grid: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.bool_],
    n_bins: int = 10,
) -> dict[str, float]:
    """Distributional calibration, after Haider et al. (2020).

    If a model's predicted survival function is right then evaluating it at
    each subject's own event time gives a Uniform(0, 1) variable -- the same
    probability integral transform used to validate the truth functions
    themselves. D-calibration bins those values and tests the counts against
    uniform.

    Censored subjects are not discarded. A subject censored at C with predicted
    survival s there is known only to fall somewhere in [0, s], so it
    contributes mass one spread across that range: the bin holding s receives
    the part of it below s, and every lower bin its full width, each divided by
    s. Dropping them would bias the statistic towards subjects who happened to
    fail early, which is precisely the subjects a censored study over-observes.

    This asks about the whole predicted distribution, where a calibration curve
    asks about one horizon. The distribution is the object this study is about.

    **Verified against the source.** The censoring weights above are the ones in
    the proof of Theorem B.3: the bucket holding ``S_c`` receives
    ``(S_c - p_k) / S_c`` and every bucket entirely below it receives
    ``(p_{k+1} - p_k) / S_c``, with uncensored subjects contributing weight one
    to their own bucket. The test is Pearson's chi-square against uniform at
    ``p > 0.05``. This implementation was originally written from a description
    and has since been checked against the paper; it agrees.

    **A precondition that matters here.** The theorem assumes survival curves
    are *strictly* monotonically decreasing. Where a curve is flat, terms in the
    proof do not cancel and buckets spanning the flat region take a larger share
    than they should -- so the statistic is inflated and the test over-rejects.
    Two of this study's four estimators, the random survival forest and gradient
    boosting, predict step functions with exactly such flat regions. Some part
    of their poorer D-calibration is therefore an artefact of the measure rather
    than evidence about the model, and the paper must say so rather than read
    the rejection at face value. The parametric estimators are unaffected: their
    curves are smooth and strictly decreasing.

    Censoring works the other way, smoothing the bucket proportions and raising
    the p-value, so the test is conservative under heavy censoring -- which is
    also why Kaplan-Meier is only *asymptotically* D-calibrated.
    """
    grid = np.asarray(grid, dtype=float)
    observed = np.asarray(time, dtype=float)
    had_event = np.asarray(event, dtype=bool)

    # Administratively censor at the end of the grid. The predicted curve is
    # only defined out to tau, so a subject observed beyond it must be treated
    # as censored there rather than have its survival clipped to S(tau): the
    # clipped version piles every late subject into one bin and rejects even a
    # model that is exactly right. Cox on a correctly specified Cox mechanism,
    # with MISE 6e-5, was rejected at p < 1e-4 before this.
    horizon = float(grid[-1])
    beyond = observed > horizon
    observed = np.where(beyond, horizon, observed)
    had_event = had_event & ~beyond

    survival_at_observed = _survival_at(predicted, grid, observed)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)

    for value in survival_at_observed[had_event]:
        index = min(int(np.searchsorted(edges, value, side="right")) - 1, n_bins - 1)
        counts[max(index, 0)] += 1.0

    width = edges[1] - edges[0]
    for value in survival_at_observed[~had_event]:
        if value <= 0.0:
            counts[0] += 1.0
            continue
        upper = min(int(np.searchsorted(edges, value, side="right")) - 1, n_bins - 1)
        upper = max(upper, 0)
        counts[upper] += (value - edges[upper]) / value
        if upper > 0:
            counts[:upper] += width / value

    total = float(counts.sum())
    if total <= 0:
        return {
            "d_calibration_statistic": float("nan"),
            "d_calibration_p": float("nan"),
        }

    expected = total / n_bins
    statistic = float(((counts - expected) ** 2 / expected).sum())

    return {
        "d_calibration_statistic": statistic,
        "d_calibration_p": float(stats.chi2.sf(statistic, df=n_bins - 1)),
        # Normalised by the sample size so it is comparable across evaluation
        # sets, which the raw chi-square is not.
        "d_calibration_normalised": statistic / total,
    }


def antolini_concordance(
    predicted: NDArray[np.float64],
    grid: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.bool_],
    max_events: int = 800,
    seed: int = 0,
) -> dict[str, float]:
    r"""Time-dependent concordance, after Antolini et al. (2005), Equation 11.

    .. math::

        C^{td} = P\bigl(\hat S(T_i \mid X_i) < \hat S(T_i \mid X_j)
                 \;\big|\; T_i < T_j,\; D_i = 1\bigr)

    Harrell and Uno reduce a predicted distribution to one score before
    comparing anything, and Sonabend et al. (2022) show the reduction itself can
    move the number. Antolini's index makes none: for a comparable pair it asks
    whether the model gave subject *i* the lower survival probability **at the
    time i actually failed**. That is the right question when hazards are not
    proportional, because the ranking of two subjects can then reverse with the
    horizon and no single score represents it.

    **Checked against the paper.** Three details were wrong when this was
    written from a description, and are now as published.

    *Comparability.* Subject *i* must have had the event; subject *j* need only
    have a later observed time, censored or not. That was already right, and the
    summary states it explicitly.

    *The horizon.* Section 2.3 restricts the index to ``[0, tau]`` by
    **administratively censoring at tau**, not by evaluating late events at the
    boundary. This previously clipped an event at ``T_i > tau`` to the last grid
    point and still counted it as an event, which both invents a comparison the
    definition excludes and evaluates it at the wrong time.

    *Ties.* Equation 12 uses a strict inequality, so a tie contributes zero, not
    the one half of Harrell's convention. This is not a technicality here: the
    random survival forest and gradient boosting predict step functions, whose
    values are frequently exactly equal, so the two conventions can differ
    materially for exactly two of the four estimators. The published definition
    is used, and ``antolini_tie_fraction`` is returned so the paper can report
    how much of the index that choice is deciding rather than leave a reader to
    wonder.

    Comparable pairs are subsampled on a fixed seed when there are many events,
    so cost does not grow quadratically across hundreds of thousands of cells.
    """
    grid = np.asarray(grid, dtype=float)
    observed = np.asarray(time, dtype=float)
    had_event = np.asarray(event, dtype=bool)

    # Administrative censoring at the end of the grid, which is tau. A subject
    # observed beyond it is censored there and is no longer an event, so it
    # cannot be the earlier member of a comparable pair.
    horizon = float(grid[-1])
    beyond = observed > horizon
    observed = np.where(beyond, horizon, observed)
    had_event = had_event & ~beyond

    event_index = np.flatnonzero(had_event)
    if event_index.size == 0:
        return {
            "c_index_antolini": float("nan"),
            "antolini_pairs": 0,
            "antolini_tie_fraction": float("nan"),
        }

    if event_index.size > max_events:
        rng = np.random.default_rng(seed)
        event_index = rng.choice(event_index, size=max_events, replace=False)

    concordant = 0.0
    tied = 0.0
    comparable = 0.0
    step_like = _step_like_rows(predicted)
    for subject in event_index:
        later = observed > observed[subject]
        if not later.any():
            continue
        event_time = float(observed[subject])
        at_event = _survival_at_common_time(predicted, grid, event_time, step_like)
        own = at_event[subject]
        others = at_event[later]
        comparable += float(others.size)
        concordant += float((own < others).sum())
        tied += float((own == others).sum())

    if comparable == 0:
        return {
            "c_index_antolini": float("nan"),
            "antolini_pairs": 0,
            "antolini_tie_fraction": float("nan"),
        }

    return {
        "c_index_antolini": concordant / comparable,
        "antolini_pairs": int(comparable),
        "antolini_tie_fraction": tied / comparable,
    }


def evaluate_all(
    *,
    predicted: NDArray[np.float64],
    truth: NDArray[np.float64],
    risk: NDArray[np.float64],
    times: NDArray[np.float64],
    tau: float,
    train_time: NDArray[np.float64],
    train_event: NDArray[np.bool_],
    eval_time: NDArray[np.float64],
    eval_event: NDArray[np.bool_],
) -> dict[str, Any]:
    """Every metric for one fitted model, on an **independent** evaluation sample.

    ``predicted``, ``truth``, ``risk`` and ``eval_*`` all refer to a fresh draw
    from the same mechanism, not to the data the model was fitted on. The
    training outcomes are still needed, but only to estimate the censoring
    distribution for the inverse-probability weights in the Brier score and
    Uno's concordance.

    Evaluating in-sample was the first design here and it was wrong. A random
    survival forest scored a Harrell concordance of 0.78 against Cox's 0.65 on
    a *correctly specified* Cox mechanism, purely by fitting noise -- which
    would have let overfitting masquerade as the discrimination-calibration gap
    this study is about. The two must be separable for the paper's claim to
    mean anything, so performance is measured out of sample throughout.
    """
    grid = np.asarray(times, dtype=float)
    results: dict[str, Any] = {}
    results.update(truth_recovery(predicted, truth, grid, tau))

    # Discrimination on one transformation applied identically to every model,
    # so the concordances are comparable. Scoring each model on its own native
    # risk is the third form of C-hacking in Sonabend et al. (2022): three
    # different mathematical objects compared with one measure.
    common = expected_mortality(predicted, grid)
    results.update(
        discrimination(common, train_time, train_event, eval_time, eval_event, tau)
    )

    # The native score, reported separately and named as such. Sonabend et al.
    # are explicit that reporting both is legitimate and only conflating them is
    # not, so these must never be pooled with the values above.
    native = discrimination(risk, train_time, train_event, eval_time, eval_event, tau)
    results.update(
        {
            "c_index_harrell_native": native["c_index_harrell"],
            "c_index_uno_native": native.get("c_index_uno", float("nan")),
        }
    )
    results.update(
        prediction_error(
            predicted, grid, train_time, train_event, eval_time, eval_event, tau
        )
    )

    index = int(np.searchsorted(grid, tau, side="right")) - 1
    results.update(
        grouped_calibration_error(
            predicted[:, index], eval_time, eval_event, float(grid[index])
        )
    )

    # Measures defined on the predicted distribution rather than on a score.
    # The contribution this study claims concerns individual survival
    # distributions, so the calibration and discrimination measures it reports
    # should be the ones defined for distributions -- not only those defined at
    # a single horizon or on a static risk ranking.
    results.update(d_calibration(predicted, grid, eval_time, eval_event))
    results.update(antolini_concordance(predicted, grid, eval_time, eval_event))

    # The same ranking taken from survival at tau alone rather than from the
    # whole curve. Under proportional hazards it agrees with the measures above;
    # where it diverges, the ranking depends on the horizon. Reported under its
    # own name because it is a third transformation, and choosing among the
    # three by which flatters a model is precisely the first form of C-hacking.
    results["c_index_at_tau"] = discrimination(
        -predicted[:, index], train_time, train_event, eval_time, eval_event, tau
    )["c_index_harrell"]

    return results
