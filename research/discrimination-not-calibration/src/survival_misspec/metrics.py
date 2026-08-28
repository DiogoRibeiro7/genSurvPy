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
    # These estimators are only defined strictly inside the observed follow-up
    # of the test set; asking outside it raises rather than extrapolating.
    upper = min(
        tau, float(np.max(test_time[np.asarray(test_event, bool)], initial=0.0))
    )
    usable = (grid > float(np.min(test_time))) & (grid < upper)

    out: dict[str, float] = {}
    if usable.sum() < 2:
        return {
            "brier_at_tau": float("nan"),
            "integrated_brier_score": float("nan"),
            "auc_mean": float("nan"),
            "prediction_error_note": "no grid point inside the usable follow-up range",
        }

    sub_grid = grid[usable]
    sub_predicted = predicted[:, usable]

    try:
        _, scores = brier_score(train, test, sub_predicted, sub_grid)
        out["brier_at_tau"] = float(scores[-1])
        out["integrated_brier_score"] = float(
            integrated_brier_score(train, test, sub_predicted, sub_grid)
        )
    except Exception as exception:  # noqa: BLE001
        out["brier_at_tau"] = float("nan")
        out["integrated_brier_score"] = float("nan")
        out["brier_error"] = f"{type(exception).__name__}: {exception}"

    try:
        auc, mean_auc = cumulative_dynamic_auc(
            train, test, 1.0 - sub_predicted, sub_grid
        )
        out["auc_mean"] = float(mean_auc)
        out["auc_at_tau"] = float(auc[-1])
    except Exception as exception:  # noqa: BLE001
        out["auc_mean"] = float("nan")
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
    results.update(
        discrimination(risk, train_time, train_event, eval_time, eval_event, tau)
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

    # The same ranking measured on predicted survival at tau rather than on the
    # model's native score. Under proportional hazards the two agree; where
    # they diverge, the ranking itself depends on the horizon, which is worth
    # recording rather than assuming away.
    results["c_index_at_tau"] = discrimination(
        -predicted[:, index], train_time, train_event, eval_time, eval_event, tau
    )["c_index_harrell"]

    return results
