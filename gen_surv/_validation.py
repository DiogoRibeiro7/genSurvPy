from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


class ValidationError(ValueError):
    """Base class for input validation errors."""


class PositiveIntegerError(ValidationError):
    """Raised when a value expected to be a positive integer is invalid."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(f"Argument '{name}' must be a positive integer; got {value!r}")


class PositiveValueError(ValidationError):
    """Raised when a value expected to be positive is invalid."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(f"Argument '{name}' must be greater than 0; got {value!r}")


class ChoiceError(ValidationError):
    """Raised when a value is not among an allowed set of choices."""

    def __init__(self, name: str, value: Any, choices: Iterable[str]) -> None:
        choices_str = "', '".join(sorted(choices))
        super().__init__(
            f"Argument '{name}' must be one of '{choices_str}'; got {value!r}"
        )


class LengthError(ValidationError):
    """Raised when a sequence does not have the expected length."""

    def __init__(self, name: str, actual: int, expected: int) -> None:
        super().__init__(
            f"Argument '{name}' must be a sequence of length {expected}; got length {actual}"
        )


class NumericSequenceError(ValidationError):
    """Raised when a sequence contains non-numeric elements."""

    def __init__(self, name: str, seq: Sequence[Any]) -> None:
        super().__init__(f"All elements in '{name}' must be numeric; got {seq!r}")


class PositiveSequenceError(ValidationError):
    """Raised when a sequence contains non-positive elements."""

    def __init__(self, name: str, seq: Sequence[Any]) -> None:
        super().__init__(
            f"All elements in '{name}' must be greater than 0; got {seq!r}"
        )


class ListOfListsError(ValidationError):
    """Raised when a value is not a list of lists."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(f"Argument '{name}' must be a list of lists; got {value!r}")


class ParameterError(ValidationError):
    """Raised when a parameter falls outside its allowed range."""

    def __init__(self, name: str, value: Any, constraint: str) -> None:
        super().__init__(f"Invalid value for '{name}': {value!r}. {constraint}")


_ALLOWED_CENSORING = {"uniform", "exponential"}


def ensure_positive_int(value: int, name: str) -> None:
    """Ensure ``value`` is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise PositiveIntegerError(name, value)


def ensure_positive(value: float | int, name: str) -> None:
    """Ensure ``value`` is a positive number."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise PositiveValueError(name, value)


def ensure_in_choices(value: str, name: str, choices: Iterable[str]) -> None:
    """Ensure ``value`` is one of the given ``choices``."""
    if value not in choices:
        raise ChoiceError(name, value, choices)


def ensure_sequence_length(seq: Sequence[Any], length: int, name: str) -> None:
    """Ensure ``seq`` has the specified ``length``."""
    if len(seq) != length:
        raise LengthError(name, len(seq), length)


def _to_float_array(seq: Sequence[Any], name: str) -> NDArray[np.float64]:
    """Convert ``seq`` to a NumPy float64 array or raise an error."""
    try:
        return np.asarray(seq, dtype=float)
    except (TypeError, ValueError) as exc:
        raise NumericSequenceError(name, seq) from exc


def ensure_numeric_sequence(seq: Sequence[Any], name: str) -> None:
    """Ensure all elements of ``seq`` are numeric."""
    _to_float_array(seq, name)


def ensure_positive_sequence(seq: Sequence[float], name: str) -> None:
    """Ensure all elements of ``seq`` are positive."""
    arr = _to_float_array(seq, name)
    if np.any(arr <= 0):
        raise PositiveSequenceError(name, seq)


def ensure_censoring_model(model_cens: str) -> None:
    """Validate that the censoring model is supported."""
    ensure_in_choices(model_cens, "model_cens", _ALLOWED_CENSORING)
