# Troubleshooting

## `ModuleNotFoundError: No module named 'gen_surv'`

The package is not installed in the interpreter you are running:

```bash
pip install gen-surv
```

From a source checkout, install it rather than relying on the working
directory:

```bash
pip install -e .      # or: poetry install
```

Check which interpreter is actually in use — a mismatch between the shell's
`python` and your editor's is the usual cause:

```python
import sys; print(sys.executable)
```

## `ImportError: cannot import name 'to_sksurv' from 'gen_surv'`

scikit-survival is the one optional dependency, and those two helpers are only
exported when it is present:

```bash
pip install scikit-survival
```

If the build fails, conda-forge has prebuilt wheels:
`conda install -c conda-forge scikit-survival`. See
[Installation](../getting-started/installation.md#the-one-optional-extra).

## `ChoiceError: Argument 'model' must be one of ...`

The model name is not one of the eleven. The message lists them all; note
`aft_ln` rather than `aft_log_normal`, and `piecewise_exponential` rather than
`piecewise`.

## `LengthError: Argument 'beta' must be a sequence of length 3`

Several models require an exact number of coefficients:

| Model | `beta` length | Also |
|---|---|---|
| `cphm` | a **scalar**, not a sequence | — |
| `cmm`, `thmm` | exactly 3 | `rate` is 6 for `cmm`, 3 for `thmm` |
| `tdcm` | exactly 2 | 3 is deprecated and warns |
| `aft_*` | any length — it sets the covariate count | — |

The error names the model it was validating, which tells you which rule
applies:

```text
LengthError: Argument 'beta' must be a sequence of length 3; got length 2.
Adjust the number of elements. (while validating inputs for model 'thmm')
```

## `DeprecationWarning: gen_tdcm uses two coefficients`

You passed three; the third has never had any effect. Drop it — see
[TDCM](../models/tdcm.md#parameters).

## `ParameterError: ... must include 'mean' and 'std'`

`covariate_params` has to be complete for its distribution; there is no
per-key defaulting. Pass all of the required keys, or pass `None` to take the
defaults. See [Covariates](../guides/covariates.md#partial-parameter-dicts-are-rejected).

## The frame has more rows than `n`

Expected for `cmm` and `thmm`: subjects contribute two or three rows each. Use
`df["id"].nunique()`, not `len(df)`. See
[Output schemas](../getting-started/schemas.md#several-rows-per-subject).

## `KeyError: 'time'` on a `tdcm` or `thmm` frame

Column names differ by model. `tdcm` has `start`/`stop` rather than `time`, and
`thmm` has `time`/`state` with no `status` at all. Most helpers accept
`time_col` and `status_col`:

```python
from gen_surv import describe_survival, generate

tdcm_df = generate(model="tdcm", n=100, dist="weibull", corr=0.5,
                   dist_par=[1.0, 2.0, 1.0, 2.0], model_cens="uniform",
                   cens_par=5.0, beta=[0.5, 0.3], lam=1.0, seed=1)

describe_survival(tdcm_df, time_col="stop")
```

## Everything is censored, or nothing is

`cens_par` is an upper bound for uniform censoring and a **mean** for
exponential censoring, so larger values censor **less**. If the event rate is
0.0 or 1.0, that dial is at the wrong end — sweep it, as in
[Censoring](../guides/censoring.md#hitting-a-target-event-rate).

## An estimator does not recover the parameter I set

Work through these in order:

1. **Sample size.** At `n=2000` a Cox estimate can miss the truth by two
   standard errors — that is ordinary variation. Repeat over seeds and look at
   the mean, or raise `n`.
2. **The parameter means what you think.** `beta` in `aft_weibull` is a log
   hazard ratio; its effect on log time is $-\beta/\texttt{shape}$. See
   [AFT models](../models/aft.md#what-beta-actually-does).
3. **The estimator matches the model.** A Fine-Gray fit will not return the
   cause-specific `betas` used to generate competing-risks data — by design.
4. **The coefficients were not random.** With `betas=None`, several models
   **draw the coefficients for you**, so there is no fixed truth to recover.
   Always pass them explicitly when validating.
5. **The layout suits the estimator.** A naive Cox fit on `tdcm` output is
   biased, and flips the sign, because the risk interval is not split at the
   crossover. Split it using the crossover time from `simulate()` — see
   [TDCM](../models/tdcm.md#analysing-it-properly).

## Results changed after upgrading

A bug fix in a sampler changes the numbers a given seed produces; 1.3.0 and
2.0.0 both did this deliberately. 3.0.0 did it for `cmm` and `thmm`
specifically, by rebuilding both on the
[multistate engine](../models/multistate.md). Pin the version alongside the
seed for anything that must reproduce. See
[Reproducibility](../getting-started/reproducibility.md#what-stability-you-can-rely-on).

## Plots do not appear

In a script or on CI there is no display. Save instead of showing:

```python
import matplotlib
matplotlib.use("Agg")

from gen_surv import generate, plot_survival_curve

df = generate(model="cphm", n=200, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=1)
fig, ax = plot_survival_curve(df)
fig.savefig("plot.png", dpi=200, bbox_inches="tight")
```

## Something else

Search or open an issue on the
[tracker](https://github.com/DiogoRibeiro7/genSurvPy/issues). A minimal snippet
including the `gen_surv.__version__` and the seed is enough to reproduce almost
anything here.
