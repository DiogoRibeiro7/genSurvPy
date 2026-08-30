"""Estimator adapters behind one interface.

Four estimators, chosen to span the assumptions that matter for this study
rather than to lengthen a benchmark table:

======================  ==========================================================
Cox proportional        Semi-parametric baseline, proportional hazards assumed.
hazards                 Correctly specified for ``cphm``, ``aft_weibull`` and
                        ``piecewise_exponential``; misspecified for the rest.
Weibull AFT             Fully parametric, monotone hazard. Correct only where the
                        truth is Weibull.
Random survival forest  Non-parametric, no proportionality assumption, but
                        estimates a discrete hazard from a finite sample.
Gradient boosted        Non-parametric with a proportional-hazards loss, so it
survival analysis       relaxes functional form while keeping proportionality.
======================  ==========================================================

Each adapter exposes ``predict_survival`` returning an ``(n, len(times))``
array and ``predict_risk`` returning a scalar score per subject, so everything
downstream is written once. Third-party API differences -- structured arrays,
step functions, DataFrames indexed by time -- stop here.

**Failure is a result, not an exception to hide.** A Cox fit that does not
converge under heavy censoring, or a forest that cannot be built at n=250, is
information about that estimator in that regime. :func:`fit_estimator` captures
the error and any warnings and returns them; nothing is silently dropped.
"""

from __future__ import annotations

import time as _time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "SurvivalEstimator",
    "FittedModel",
    "ADAPTERS",
    "build_estimator",
    "fit_estimator",
]


class SurvivalEstimator(Protocol):
    """What the rest of the study is allowed to assume about an estimator."""

    def fit(
        self,
        X: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.bool_],
    ) -> None: ...

    def predict_survival(
        self, X: NDArray[np.float64], times: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def predict_survival_at_times(
        self, X: NDArray[np.float64], times: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def predict_risk(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def coefficients(self) -> dict[str, float] | None: ...


def _frame(X: NDArray[np.float64]) -> pd.DataFrame:
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return pd.DataFrame(matrix, columns=[f"X{i}" for i in range(matrix.shape[1])])


def _structured(time: NDArray[np.float64], event: NDArray[np.bool_]) -> NDArray[Any]:
    """scikit-survival's ``(event, time)`` structured array."""
    out = np.empty(len(time), dtype=[("event", bool), ("time", float)])
    out["event"] = np.asarray(event, dtype=bool)
    out["time"] = np.asarray(time, dtype=float)
    return out


def _clip_survival(surface: NDArray[np.float64]) -> NDArray[np.float64]:
    """Predictions must be probabilities, and must not increase in time.

    Estimators occasionally return values a hair outside ``[0, 1]`` from
    floating-point work, and step-function interpolation can produce a tiny
    upward step. Both are numerical, not substantive, but a survival curve that
    rises makes the integrated losses meaningless, so it is enforced here
    rather than assumed.
    """
    clipped = np.clip(surface, 0.0, 1.0)
    return np.minimum.accumulate(clipped, axis=1)


# ---------------------------------------------------------------------------
# lifelines
# ---------------------------------------------------------------------------


class CoxPHAdapter:
    """Cox proportional hazards, via ``lifelines``."""

    def __init__(self, **params: Any) -> None:
        from lifelines import CoxPHFitter

        self._model = CoxPHFitter(**params)
        self._columns: list[str] = []

    def fit(self, X, time, event) -> None:  # type: ignore[no-untyped-def]
        frame = _frame(X)
        self._columns = list(frame.columns)
        frame = frame.assign(
            _time=np.asarray(time, float), _event=np.asarray(event, int)
        )
        self._model.fit(frame, duration_col="_time", event_col="_event")

    def predict_survival(self, X, times):  # type: ignore[no-untyped-def]
        grid = np.asarray(times, dtype=float)
        predicted = self._model.predict_survival_function(_frame(X), times=grid)
        # lifelines returns times as the index and subjects as columns.
        return _clip_survival(predicted.to_numpy().T)

    def predict_survival_at_times(self, X, times):  # type: ignore[no-untyped-def]
        grid = np.unique(np.asarray(times, dtype=float))
        surface = self.predict_survival(X, grid)
        columns = np.searchsorted(grid, np.asarray(times, dtype=float), side="left")
        return surface[np.arange(surface.shape[0]), columns]

    def predict_risk(self, X):  # type: ignore[no-untyped-def]
        return np.asarray(
            self._model.predict_partial_hazard(_frame(X)), dtype=float
        ).ravel()

    def coefficients(self) -> dict[str, float] | None:
        return {name: float(value) for name, value in self._model.params_.items()}


class WeibullAFTAdapter:
    """Weibull accelerated failure time, via ``lifelines``."""

    def __init__(self, **params: Any) -> None:
        from lifelines import WeibullAFTFitter

        self._model = WeibullAFTFitter(**params)

    def fit(self, X, time, event) -> None:  # type: ignore[no-untyped-def]
        frame = _frame(X).assign(
            _time=np.asarray(time, float), _event=np.asarray(event, int)
        )
        self._model.fit(frame, duration_col="_time", event_col="_event")

    def predict_survival(self, X, times):  # type: ignore[no-untyped-def]
        grid = np.asarray(times, dtype=float)
        predicted = self._model.predict_survival_function(_frame(X), times=grid)
        return _clip_survival(predicted.to_numpy().T)

    def predict_survival_at_times(self, X, times):  # type: ignore[no-untyped-def]
        grid = np.unique(np.asarray(times, dtype=float))
        surface = self.predict_survival(X, grid)
        columns = np.searchsorted(grid, np.asarray(times, dtype=float), side="left")
        return surface[np.arange(surface.shape[0]), columns]

    def predict_risk(self, X):  # type: ignore[no-untyped-def]
        # Higher risk must mean earlier failure, so negate the expected time.
        expectation = np.asarray(
            self._model.predict_expectation(_frame(X)), dtype=float
        ).ravel()
        return -expectation

    def coefficients(self) -> dict[str, float] | None:
        params = self._model.params_
        return {
            f"{level}:{name}": float(value) for (level, name), value in params.items()
        }


# ---------------------------------------------------------------------------
# scikit-survival
# ---------------------------------------------------------------------------


class _SksurvAdapter:
    """Shared plumbing: step functions evaluated on our own grid."""

    _model: Any

    def fit(self, X, time, event) -> None:  # type: ignore[no-untyped-def]
        self._model.fit(_frame(X).to_numpy(), _structured(time, event))

    def predict_survival(self, X, times):  # type: ignore[no-untyped-def]
        grid = np.asarray(times, dtype=float)
        functions = self._model.predict_survival_function(_frame(X).to_numpy())

        surface = np.empty((len(functions), grid.size), dtype=float)
        for row, function in enumerate(functions):
            # Outside the observed support a step function is undefined; carry
            # the last estimated value forward, which is the usual convention
            # and the only one that keeps the curve non-increasing.
            support = np.clip(grid, function.x[0], function.x[-1])
            surface[row] = function(support)
            surface[row][grid < function.x[0]] = 1.0

        return _clip_survival(surface)

    def predict_survival_functions(self, X):  # type: ignore[no-untyped-def]
        """Native scikit-survival step functions for exact-time metrics."""
        return self._model.predict_survival_function(_frame(X).to_numpy())

    def predict_survival_at_times(self, X, times):  # type: ignore[no-untyped-def]
        functions = self.predict_survival_functions(X)
        out = np.empty(len(functions), dtype=float)
        for row, (function, time) in enumerate(
            zip(functions, np.asarray(times, float))
        ):
            if time < function.x[0]:
                out[row] = 1.0
            else:
                out[row] = float(function(min(float(time), float(function.x[-1]))))
        return np.clip(out, 0.0, 1.0)

    def predict_risk(self, X):  # type: ignore[no-untyped-def]
        return np.asarray(
            self._model.predict(_frame(X).to_numpy()), dtype=float
        ).ravel()

    def coefficients(self) -> dict[str, float] | None:
        return None  # no coefficients corresponding to DGP parameters


class RandomSurvivalForestAdapter(_SksurvAdapter):
    """Random survival forest, via ``scikit-survival``."""

    def __init__(self, **params: Any) -> None:
        from sksurv.ensemble import RandomSurvivalForest

        params.setdefault("random_state", 0)
        self._model = RandomSurvivalForest(**params)


class GradientBoostedAdapter(_SksurvAdapter):
    """Gradient boosted survival analysis, via ``scikit-survival``."""

    def __init__(self, **params: Any) -> None:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        params.setdefault("random_state", 0)
        self._model = GradientBoostingSurvivalAnalysis(**params)


ADAPTERS: dict[str, type] = {
    "cox_ph": CoxPHAdapter,
    "weibull_aft": WeibullAFTAdapter,
    "random_survival_forest": RandomSurvivalForestAdapter,
    "gradient_boosted": GradientBoostedAdapter,
}


def build_estimator(adapter: str, params: Mapping[str, Any] | None = None) -> Any:
    if adapter not in ADAPTERS:
        raise KeyError(
            f"unknown estimator adapter {adapter!r}; available: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[adapter](**dict(params or {}))


@dataclass
class FittedModel:
    """The outcome of one fit, successful or not."""

    estimator_id: str
    fitted: bool
    runtime_seconds: float
    model: Any = None
    error: str = ""
    error_type: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return not self.fitted


def fit_estimator(
    estimator_id: str,
    adapter: str,
    params: Mapping[str, Any],
    X: NDArray[np.float64],
    time: NDArray[np.float64],
    event: NDArray[np.bool_],
) -> FittedModel:
    """Fit, capturing failure and warnings instead of propagating them.

    A convergence failure under heavy censoring is a property of the estimator
    in that regime and belongs in the results, so this never raises for a
    modelling failure. It does record the exception type and message, which is
    what makes the failure table in the paper interpretable rather than a bare
    count.
    """
    started = _time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = build_estimator(adapter, params)
            model.fit(X, time, event)
            fitted, error, error_type = True, "", ""
        except Exception as exception:  # noqa: BLE001 - failure is the result
            model = None
            fitted = False
            error = f"{exception}".strip() or traceback.format_exc(limit=1).strip()
            error_type = type(exception).__name__

    return FittedModel(
        estimator_id=estimator_id,
        fitted=fitted,
        runtime_seconds=_time.perf_counter() - started,
        model=model,
        error=error,
        error_type=error_type,
        warnings=sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
    )
