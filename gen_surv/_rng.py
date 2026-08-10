"""Shared random-number-generator resolution.

Every simulator in :mod:`gen_surv` takes a ``seed`` argument and turns it into a
:class:`numpy.random.Generator` with :func:`resolve_rng`. No simulator draws
from the global NumPy random state, so results are reproducible from the
``seed`` alone and independent of anything else the process has drawn.

Passing an existing generator lets several simulators share one stream, which
is what makes composed simulations reproducible as a whole.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.random import Generator

#: Anything the simulators accept as a source of randomness.
RandomStateLike: TypeAlias = int | Generator | None


def resolve_rng(seed: RandomStateLike = None) -> Generator:
    """Return a NumPy generator for ``seed``.

    Parameters
    ----------
    seed : int or numpy.random.Generator, optional
        An ``int`` seeds a fresh generator, a :class:`numpy.random.Generator`
        is returned unchanged so callers can share one stream, and ``None``
        produces a fresh unseeded generator.

    Returns
    -------
    numpy.random.Generator
        The generator to draw from.

    Examples
    --------
    Equal seeds give equal draws:

    >>> from gen_surv._rng import resolve_rng
    >>> float(resolve_rng(42).uniform()) == float(resolve_rng(42).uniform())
    True

    An existing generator is passed straight through:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> resolve_rng(rng) is rng
    True
    """
    if isinstance(seed, Generator):
        return seed

    return np.random.default_rng(seed)
