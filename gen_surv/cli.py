"""
Command-line interface for gen_surv.

This module provides a command-line interface for generating survival data
using the gen_surv package.
"""

from typing import Optional, List, Tuple
import typer
from gen_surv.interface import generate

app = typer.Typer(help="Generate synthetic survival datasets.")


@app.command()
def dataset(
    model: str = typer.Argument(
        ..., 
        help="Model to simulate [cphm, cmm, tdcm, thmm, aft_ln, aft_weibull]"
    ),
    n: int = typer.Option(100, help="Number of samples"),
    model_cens: str = typer.Option(
        "uniform", help="Censoring model: 'uniform' or 'exponential'"
    ),
    cens_par: float = typer.Option(1.0, help="Censoring parameter"),
    beta: List[float] = typer.Option(
        [0.5], help="Regression coefficient(s). Provide multiple values for multi-parameter models."
    ),
    covar: Optional[float] = typer.Option(
        2.0, help="Covariate range (for CPHM, CMM, THMM)"
    ),
    sigma: Optional[float] = typer.Option(
        1.0, help="Standard deviation parameter (for log-normal AFT)"
    ),
    shape: Optional[float] = typer.Option(
        1.5, help="Shape parameter (for Weibull AFT)"
    ),
    scale: Optional[float] = typer.Option(
        2.0, help="Scale parameter (for Weibull AFT)"
    ),
    seed: Optional[int] = typer.Option(
        None, help="Random seed for reproducibility"
    ),
    output: Optional[str] = typer.Option(
        None, "-o", help="Output CSV file. Prints to stdout if omitted."
    ),
) -> None:
    """Generate survival data and optionally save to CSV.

    Examples:
        # Generate data from CPHM model
        $ gen_surv dataset cphm --n 100 --beta 0.5 --covar 2.0 -o cphm_data.csv

        # Generate data from Weibull AFT model
        $ gen_surv dataset aft_weibull --n 200 --beta 0.5 --beta -0.3 --shape 1.5 --scale 2.0 -o aft_data.csv
    """
    # Prepare arguments based on the selected model
    kwargs = {
        "model": model,
        "n": n,
        "model_cens": model_cens,
        "cens_par": cens_par,
        "seed": seed
    }
    
    # Add model-specific parameters
    if model in ["cphm", "cmm", "thmm"]:
        # These models use a single beta and covar
        kwargs["beta"] = beta[0] if len(beta) > 0 else 0.5
        kwargs["covar"] = covar
        
    elif model == "aft_ln":
        # Log-normal AFT model uses beta list and sigma
        kwargs["beta"] = beta
        kwargs["sigma"] = sigma
        
    elif model == "aft_weibull":
        # Weibull AFT model uses beta list, shape, and scale
        kwargs["beta"] = beta
        kwargs["shape"] = shape
        kwargs["scale"] = scale
    
    # Generate the data
    df = generate(**kwargs)
    
    # Output the data
    if output:
        df.to_csv(output, index=False)
        typer.echo(f"Saved dataset to {output}")
    else:
        typer.echo(df.to_csv(index=False))


@app.command()
def visualize(
    input_file: str = typer.Argument(
        ..., help="Input CSV file containing survival data"
    ),
    time_col: str = typer.Option(
        "time", help="Column containing time/duration values"
    ),
    status_col: str = typer.Option(
        "status", help="Column containing event indicator (1=event, 0=censored)"
    ),
    group_col: Optional[str] = typer.Option(
        None, help="Column to use for stratification"
    ),
    output: str = typer.Option(
        "survival_plot.png", help="Output image file"
    ),
) -> None:
    """Visualize survival data from a CSV file.
    
    Examples:
        # Generate a Kaplan-Meier plot from a CSV file
        $ gen_surv visualize data.csv --time-col time --status-col status -o km_plot.png
        
        # Generate a stratified plot using a grouping variable
        $ gen_surv visualize data.csv --group-col X0 -o stratified_plot.png
    """
    try:
        import pandas as pd
        from gen_surv.visualization import plot_survival_curve
        import matplotlib.pyplot as plt
    except ImportError:
        typer.echo(
            "Error: Visualization requires matplotlib and lifelines. "
            "Install them with: pip install matplotlib lifelines"
        )
        raise typer.Exit(1)
    
    # Load the data
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        typer.echo(f"Error loading CSV file: {str(e)}")
        raise typer.Exit(1)
    
    # Check required columns
    if time_col not in data.columns:
        typer.echo(f"Error: Time column '{time_col}' not found in data")
        raise typer.Exit(1)
    
    if status_col not in data.columns:
        typer.echo(f"Error: Status column '{status_col}' not found in data")
        raise typer.Exit(1)
    
    if group_col is not None and group_col not in data.columns:
        typer.echo(f"Error: Group column '{group_col}' not found in data")
        raise typer.Exit(1)
    
    # Create the plot
    fig, ax = plot_survival_curve(
        data=data,
        time_col=time_col,
        status_col=status_col,
        group_col=group_col
    )
    
    # Save the plot
    plt.savefig(output, dpi=300, bbox_inches="tight")
    typer.echo(f"Plot saved to {output}")


if __name__ == "__main__":
    app()
