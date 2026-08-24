from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd

from .interface import ModelType, generate
from .validation import ensure_in_choices


class BaseEstimatorProto(Protocol):
    """Protocol capturing the minimal scikit-learn estimator interface."""

    def get_params(self, deep: bool = ...) -> dict[str, object]: ...

    def set_params(self, **params: object) -> "BaseEstimatorProto": ...


if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from sklearn.base import BaseEstimator as SklearnBase
else:  # pragma: no cover - runtime import with fallback
    try:
        from sklearn.base import BaseEstimator as SklearnBase
    except Exception:

        class SklearnBase:  # noqa: D401 - simple runtime stub
            """Minimal stub if scikit-learn is not installed."""

            def get_params(self, deep: bool = True) -> dict[str, object]:
                return {}

            def set_params(self, **params: object) -> "SklearnBase":
                return self


class GenSurvDataGenerator(SklearnBase, BaseEstimatorProto):
    """Scikit-learn compatible wrapper around :func:`gen_surv.generate`."""

    def __init__(
        self, model: ModelType, return_type: str = "df", **kwargs: object
    ) -> None:
        ensure_in_choices(return_type, "return_type", {"df", "dict"})
        self.model = model
        self.return_type = return_type
        self.kwargs = kwargs

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return every parameter, including the ones forwarded to the model.

        scikit-learn builds this by introspecting ``__init__``, which cannot see
        through ``**kwargs``. Without the override, ``clone`` -- and therefore
        every pipeline, ``GridSearchCV`` and ``cross_val_score`` -- would drop
        the model's parameters and produce an estimator that fails on use.

        Parameters
        ----------
        deep : bool
            Accepted for interface compatibility. There are no nested
            estimators, so it makes no difference.

        Returns
        -------
        dict[str, object]
            ``model`` and ``return_type`` together with the model's own
            arguments.
        """
        return {"model": self.model, "return_type": self.return_type, **self.kwargs}

    def set_params(self, **params: object) -> "GenSurvDataGenerator":
        """Set parameters, whether they belong to the wrapper or the model.

        Parameters
        ----------
        **params
            Any of ``model``, ``return_type``, or an argument of the wrapped
            generator.

        Returns
        -------
        GenSurvDataGenerator
            ``self``, as scikit-learn expects.
        """
        if "model" in params:
            self.model = cast("ModelType", params.pop("model"))
        if "return_type" in params:
            return_type = params.pop("return_type")
            ensure_in_choices(cast(str, return_type), "return_type", {"df", "dict"})
            self.return_type = cast(str, return_type)

        self.kwargs = {**self.kwargs, **params}
        return self

    def fit(
        self, X: pd.DataFrame | None = None, y: pd.Series | None = None
    ) -> "GenSurvDataGenerator":
        return self

    def transform(
        self, X: pd.DataFrame | None = None
    ) -> pd.DataFrame | dict[str, list[object]]:
        df = generate(self.model, **self.kwargs)
        if self.return_type == "df":
            return df
        if self.return_type == "dict":
            return df.to_dict(orient="list")
        raise AssertionError("Unreachable due to validation")

    def fit_transform(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        **fit_params: object,
    ) -> pd.DataFrame | dict[str, list[object]]:
        return self.fit(X, y).transform(X)
