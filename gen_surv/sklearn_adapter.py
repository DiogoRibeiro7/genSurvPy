from __future__ import annotations

from typing import Any, Optional

from .interface import generate

try:  # pragma: no cover - only imported if sklearn is installed
    from sklearn.base import BaseEstimator
except Exception:  # pragma: no cover - fallback when sklearn missing
    class BaseEstimator:  # type: ignore
        """Minimal stub if scikit-learn is not installed."""


class GenSurvDataGenerator(BaseEstimator):
    """Scikit-learn compatible wrapper around :func:`gen_surv.generate`."""

    def __init__(self, model: str, return_type: str = "df", **kwargs: Any) -> None:
        self.model = model
        self.return_type = return_type
        self.kwargs = kwargs

    def fit(self, X: Optional[Any] = None, y: Optional[Any] = None) -> "GenSurvDataGenerator":
        return self

    def transform(self, X: Optional[Any] = None) -> Any:
        df = generate(self.model, **self.kwargs)
        if self.return_type == "df":
            return df
        if self.return_type == "dict":
            return df.to_dict(orient="list")
        raise ValueError("return_type must be 'df' or 'dict'")

    def fit_transform(self, X: Optional[Any] = None, y: Optional[Any] = None, **fit_params: Any) -> Any:
        return self.fit(X, y).transform(X)
