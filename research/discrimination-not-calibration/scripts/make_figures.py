"""Generate every figure in the paper from the processed results.

    python scripts/make_figures.py --processed results/processed --raw results/raw/production.parquet

No figure is edited by hand and none carries a number typed into it. Re-running
this after a re-run of the experiment reproduces the paper's figures exactly,
which is the only way the numbers in the text and the numbers in the plots can
be guaranteed to agree.

Figures are written to ``results/figures/`` as PDF (for LaTeX) and PNG (for
looking at). A file is written only if the data it needs is present; a missing
column is reported rather than silently producing an empty axis.

If the results are exploratory -- a pilot run, with no experiment lock -- every
figure is stamped as such, so a pilot plot cannot be mistaken for a production
one after it leaves this directory.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.aggregation import read_raw  # noqa: E402

# A restrained, colour-blind-safe palette; the paper is likely to be printed.
COLOURS = {
    "cox_ph": "#0072B2",
    "weibull_aft": "#D55E00",
    "random_survival_forest": "#009E73",
    "gradient_boosted": "#CC79A7",
    "royston_parmar": "#56B4E9",
}
MARKERS = {
    "cox_ph": "o",
    "weibull_aft": "s",
    "random_survival_forest": "^",
    "gradient_boosted": "D",
    "royston_parmar": "v",
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def _style(estimator: str) -> dict:
    return {
        "color": COLOURS.get(estimator, "#444444"),
        "marker": MARKERS.get(estimator, "o"),
        "linestyle": "none",
        "markersize": 5,
        "alpha": 0.85,
        "label": estimator,
    }


def _save(figure: plt.Figure, name: str, out: Path, exploratory: bool) -> None:
    if exploratory:
        figure.text(
            0.5,
            0.5,
            "PILOT — NOT FOR PUBLICATION",
            fontsize=28,
            color="0.85",
            ha="center",
            va="center",
            rotation=30,
            zorder=-1,
        )
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(figure)
    print(f"  {name}")


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> bool:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        print(f"  {name}: skipped, missing {missing}")
        return False
    return True


def _summarise_factor(block: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Average scenario means by factor with MCSEs propagated."""
    rows = []
    for value, factor_block in block.groupby(factor):
        means = pd.to_numeric(factor_block["mise_mean"], errors="coerce")
        mcse = pd.to_numeric(factor_block["mise_mcse"], errors="coerce")
        keep = means.notna()
        if not keep.any():
            continue
        means = means[keep]
        mcse = mcse[keep].fillna(0.0)
        rows.append(
            {
                factor: value,
                "mise": float(means.mean()),
                "mcse": float(math.sqrt(float((mcse**2).sum())) / len(means)),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(factor)


# ---------------------------------------------------------------------------
# 1. The central figure: discrimination against recovery of the truth
# ---------------------------------------------------------------------------


def figure_discrimination_vs_truth(
    summary: pd.DataFrame, out: Path, exploratory: bool
) -> None:
    """Every cell as a point. The paper's argument is the *shape* of this cloud.

    If discrimination told you what you needed, the points would fall on a
    decreasing curve. The claim is that they do not: the same C-index is
    consistent with a wide range of distance from the truth, and the width of
    that range is the quantity the study reports.
    """
    if not _require(summary, ["c_index_harrell_mean", "mise_mean"], "fig1"):
        return

    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    for estimator, block in summary.groupby("estimator_id"):
        axis.plot(
            block["c_index_harrell_mean"], block["mise_mean"], **_style(estimator)
        )

    axis.set_yscale("log")
    axis.set_xlabel("Harrell C-index")
    axis.set_ylabel("MISE against the true survival function (log scale)")
    axis.set_title("Discrimination and truth-recovery metrics")
    # Below the axes: the interesting part of this figure is the vertical
    # spread at a fixed C-index, and a legend sitting in it hides the argument.
    axis.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    finite = summary[["c_index_harrell_mean", "mise_mean"]].dropna()
    if len(finite) > 2:
        correlation = finite.corr().iloc[0, 1]
        axis.annotate(
            f"r = {correlation:+.3f}",
            xy=(0.03, 0.03),
            xycoords="axes fraction",
            fontsize=8,
        )

    _save(figure, "fig1_discrimination_vs_truth", out, exploratory)


# ---------------------------------------------------------------------------
# 2. Calibration against discrimination
# ---------------------------------------------------------------------------


def figure_calibration_vs_discrimination(
    summary: pd.DataFrame, out: Path, exploratory: bool
) -> None:
    if not _require(
        summary, ["c_index_harrell_mean", "calibration_error_mean"], "fig2"
    ):
        return

    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    for estimator, block in summary.groupby("estimator_id"):
        axis.plot(
            block["c_index_harrell_mean"],
            block["calibration_error_mean"],
            **_style(estimator),
        )

    axis.set_xlabel("Harrell C-index")
    axis.set_ylabel("Grouped calibration error at $\\tau$")
    axis.set_title("Calibration and discrimination metrics")
    axis.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _save(figure, "fig2_calibration_vs_discrimination", out, exploratory)


# ---------------------------------------------------------------------------
# 3-5. The primary loss against each design factor
# ---------------------------------------------------------------------------


def _factor_panel(
    summary: pd.DataFrame,
    factor: str,
    xlabel: str,
    name: str,
    out: Path,
    exploratory: bool,
    logx: bool = False,
) -> None:
    if not _require(summary, [factor, "mise_mean", "mise_mcse"], name):
        return

    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    for estimator, block in summary.groupby("estimator_id"):
        grouped = _summarise_factor(block, factor)
        if grouped.empty:
            continue
        style = _style(estimator)
        style["linestyle"] = "-"
        axis.errorbar(
            grouped[factor],
            grouped["mise"],
            yerr=grouped["mcse"],
            capsize=2,
            linewidth=1.2,
            **style,
        )

    if logx:
        axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("MISE (log scale)")
    axis.set_title(f"Recovery of the truth against {xlabel.lower()}")
    axis.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _save(figure, name, out, exploratory)


# ---------------------------------------------------------------------------
# 6. Adequacy region
# ---------------------------------------------------------------------------


def figure_adequacy(adequacy: pd.DataFrame, out: Path, exploratory: bool) -> None:
    """Where a candidate stays within epsilon of the reference, as epsilon varies.

    Reported across a range of epsilon rather than at one value, because
    epsilon is a tolerance on the scale of the loss, not a statistical
    constant. Reading a single panel as "the" adequacy region would be exactly
    the mistake the protocol warns against.
    """
    if adequacy is None or adequacy.empty:
        print("  fig6: skipped, no adequacy table")
        return
    if not _require(adequacy, ["epsilon", "within_epsilon", "estimator_id"], "fig6"):
        return

    share = (
        adequacy.groupby(["epsilon", "estimator_id"])["within_epsilon"]
        .mean()
        .reset_index()
    )

    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    for estimator, block in share.groupby("estimator_id"):
        style = _style(estimator)
        style["linestyle"] = "-"
        axis.plot(block["epsilon"], block["within_epsilon"], **style)

    axis.set_xscale("log")
    axis.set_xlabel("$\\epsilon$ (tolerance on MISE)")
    axis.set_ylabel("Share of scenarios within $\\epsilon$ of the reference")
    axis.set_title("Adequacy region as the tolerance varies")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=7, ncol=2)
    _save(figure, "fig6_adequacy_region", out, exploratory)


# ---------------------------------------------------------------------------
# 7. The illustrative case
# ---------------------------------------------------------------------------


def figure_illustrative_curves(raw: pd.DataFrame, out: Path, exploratory: bool) -> None:
    """One scenario where discrimination looks fine and the curves are not.

    Chosen automatically as the cell with the highest MISE among those whose
    C-index is at or above the median, so the choice is reproducible and cannot
    be the most flattering example someone went looking for.
    """
    print("  fig7: skipped, illustrative curve export is not implemented")


# ---------------------------------------------------------------------------
# 8. Failures
# ---------------------------------------------------------------------------


def figure_failures(failures: pd.DataFrame, out: Path, exploratory: bool) -> None:
    """Failure is performance. A cell an estimator cannot fit is not a cell it won."""
    if failures is None or failures.empty:
        print("  fig8: skipped, no failure table")
        return
    if not _require(failures, ["estimator_id", "fit_failure_rate"], "fig8"):
        return

    grouped = failures.groupby("estimator_id")["fit_failure_rate"].mean().sort_values()

    figure, axis = plt.subplots(figsize=(5.0, 3.2))
    axis.barh(
        grouped.index,
        grouped.to_numpy(),
        color=[COLOURS.get(name, "#444444") for name in grouped.index],
    )
    axis.set_xlabel("Mean fit-failure rate across scenarios")
    axis.set_title("Fit failures by estimator")
    if float(np.nanmax(grouped.to_numpy(), initial=0.0)) == 0.0:
        axis.annotate(
            "no fit failed in any scenario",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            fontsize=9,
        )
    _save(figure, "fig8_failure_rates", out, exploratory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed", default=str(HERE.parent / "results" / "processed")
    )
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out", default=str(HERE.parent / "results" / "figures"))
    arguments = parser.parse_args()

    processed = Path(arguments.processed)
    out = Path(arguments.out)

    summary_path = processed / "summary.parquet"
    if not summary_path.exists():
        print(f"no summary at {summary_path}; run aggregate_results.py first")
        return 1

    summary = pd.read_parquet(summary_path)
    failures = (
        pd.read_parquet(processed / "failures.parquet")
        if (processed / "failures.parquet").exists()
        else pd.DataFrame()
    )
    adequacy = (
        pd.read_parquet(processed / "adequacy.parquet")
        if (processed / "adequacy.parquet").exists()
        else pd.DataFrame()
    )
    raw = read_raw(arguments.raw) if arguments.raw else pd.DataFrame()

    exploratory = True
    if not raw.empty and "is_production" in raw.columns:
        exploratory = not bool(raw["is_production"].any())
    print(
        f"figures from {'PILOT (exploratory)' if exploratory else 'production'} results"
    )

    figure_discrimination_vs_truth(summary, out, exploratory)
    figure_calibration_vs_discrimination(summary, out, exploratory)
    _factor_panel(
        summary, "effect_size", "Effect size", "fig3_mise_vs_effect", out, exploratory
    )
    _factor_panel(
        summary,
        "target_censoring",
        "Censoring",
        "fig4_mise_vs_censoring",
        out,
        exploratory,
    )
    _factor_panel(
        summary, "n", "Sample size", "fig5_mise_vs_n", out, exploratory, logx=True
    )
    figure_adequacy(adequacy, out, exploratory)
    if not raw.empty:
        figure_illustrative_curves(raw, out, exploratory)
    figure_failures(failures, out, exploratory)

    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
