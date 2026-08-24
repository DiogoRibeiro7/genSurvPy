"""Configuration and ground truth alongside the simulated data.

A generator returns a ``DataFrame``, which is what an analyst wants and what an
estimator consumes. It is not what a *methodologist* wants: the interesting
quantities in a simulation study are the ones a real dataset could never
contain — the coefficients that produced it, the latent event time before
censoring intervened, which subjects are cured, when a covariate crossed over.

:func:`simulate` returns those alongside the frame:

>>> from gen_surv import simulate
>>> result = simulate("cphm", n=100, beta=0.5, covariate_range=2.0,
...                   model_cens="uniform", cens_par=1.0, seed=42)
>>> result.data.shape
(100, 3)
>>> sorted(result.truth)
['beta', 'censoring_time', 'covariates', 'event_time', 'linear_predictor']

The existing ``gen_*`` functions are unchanged and still return a frame. This
is an addition, not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from .validation import ParameterError

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "simulate",
]


@dataclass(frozen=True)
class SimulationConfig:
    """Everything needed to reproduce a simulated dataset.

    Attributes
    ----------
    model : str
        The generator's registered name, as passed to :func:`gen_surv.generate`.
    params : Mapping[str, Any]
        The keyword arguments it was called with, seed included.
    version : str
        The ``gen_surv`` version that produced the data. A bug fix in a sampler
        changes what a seed produces, so the version is part of the
        specification and not decoration.
    """

    model: str
    params: Mapping[str, Any] = field(default_factory=dict)
    version: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ParameterError("model", self.model, "must be a non-empty string")
        object.__setattr__(self, "params", dict(self.params))
        if not self.version:
            from . import __version__

            object.__setattr__(self, "version", __version__)

    @property
    def seed(self) -> Any:
        """The seed the data was produced with, or ``None`` if unseeded."""
        return self.params.get("seed")

    def replace(self, **changes: Any) -> "SimulationConfig":
        """Return a copy with ``changes`` applied to the parameters.

        The natural way to sweep: hold a scenario fixed and vary one thing.

        >>> base = SimulationConfig("cphm", {"n": 100, "beta": 0.5})
        >>> base.replace(seed=7).params["seed"]
        7
        >>> base.params.get("seed") is None      # the original is untouched
        True
        """
        return SimulationConfig(
            model=self.model, params={**self.params, **changes}, version=self.version
        )

    def run(self) -> "SimulationResult":
        """Run this configuration and return the result."""
        return simulate(self.model, **self.params)

    def to_dict(self) -> dict[str, Any]:
        """A plain dictionary, for writing to JSON or YAML."""
        return {
            "model": self.model,
            "params": dict(self.params),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationConfig":
        """Rebuild a configuration from :meth:`to_dict` output."""
        return cls(
            model=payload["model"],
            params=payload.get("params", {}),
            version=payload.get("version", ""),
        )


@dataclass(frozen=True)
class SimulationResult:
    """Simulated data, the configuration behind it, and the ground truth.

    Attributes
    ----------
    data : pandas.DataFrame
        Exactly what the corresponding ``gen_*`` function returns.
    config : SimulationConfig
        The call that produced it.
    truth : Mapping[str, Any]
        Quantities a real dataset could not contain. Which keys are present
        depends on the model; see :func:`simulate`.
    """

    data: pd.DataFrame
    config: SimulationConfig
    truth: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "truth", dict(self.truth))

    def __len__(self) -> int:
        return len(self.data)

    @property
    def n_subjects(self) -> int:
        """Number of subjects, which is not ``len(data)`` for every model."""
        if "id" in self.data.columns:
            return int(self.data["id"].nunique())
        return len(self.data)

    def truth_frame(self) -> pd.DataFrame:
        """The per-subject entries of ``truth`` as a frame.

        Scalars and anything not one-per-subject are left out, so the result
        lines up with the subjects and can be joined onto the data.
        """
        n = self.n_subjects
        columns = {
            key: np.asarray(value)
            for key, value in self.truth.items()
            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == n
        }
        return pd.DataFrame(columns)

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"SimulationResult(model={self.config.model!r}, "
            f"rows={len(self.data)}, subjects={self.n_subjects}, "
            f"truth={sorted(self.truth)})"
        )


def simulate(model: str, **kwargs: Any) -> SimulationResult:
    """Generate data and return it with its configuration and ground truth.

    Parameters
    ----------
    model : str
        Any name accepted by :func:`gen_surv.generate`.
    **kwargs
        The model's parameters, exactly as for :func:`gen_surv.generate`.

    Returns
    -------
    SimulationResult
        The frame, the configuration, and whatever ground truth the model can
        expose.

    Notes
    -----
    The keys in ``truth`` vary by model. Common ones:

    ``beta`` / ``betas``
        The coefficients actually used. This matters most where they were
        **drawn at random** because the caller omitted them, in which case
        there is otherwise no way to learn what they were.
    ``covariates``
        The covariate matrix, as an array.
    ``linear_predictor``
        ``covariates @ betas``.
    ``event_time`` and ``censoring_time``
        The latent times before the minimum of the two was taken, so you can
        see what censoring hid.

    Model-specific keys are documented on each model's page. A generator that
    cannot expose anything beyond its frame returns an empty ``truth`` rather
    than inventing entries.

    Examples
    --------
    >>> from gen_surv import simulate
    >>> result = simulate("piecewise_exponential", n=50, breakpoints=[1.0],
    ...                   hazard_rates=[0.5, 1.5], seed=7)
    >>> result.truth["betas"]        # drawn at random, and otherwise unknowable
    array([...])
    """
    from ._truth import capture
    from .interface import ModelType, generate

    config = SimulationConfig(model=model, params=dict(kwargs))

    # `generate` validates the name against the registry and raises a
    # ChoiceError listing the valid ones, so the cast only tells the type
    # checker what that check already guarantees at runtime.
    with capture() as truth:
        data = generate(cast(ModelType, model), **kwargs)

    return SimulationResult(data=data, config=config, truth=truth)
