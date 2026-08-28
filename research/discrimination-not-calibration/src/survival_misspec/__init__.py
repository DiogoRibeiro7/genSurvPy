"""Truth-based evaluation of survival distribution predictions.

Research code for the paper of that name. It uses ``gen_surv`` as its
simulation engine and lives inside the same repository, but it is not part of
the published package and nothing here is imported by ``gen_surv``.

The dependency runs one way only: research imports package.
"""

__all__ = ["__version__"]

#: Protocol version for this study, independent of the ``gen_surv`` version.
#: Incremented when the experimental design changes materially, which starts a
#: new experiment rather than amending an old one.
__version__ = "0.1.0"
