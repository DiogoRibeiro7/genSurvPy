import csv
from typing import Optional
import typer
from gen_surv.interface import generate

app = typer.Typer(help="Generate synthetic survival datasets.")

@app.command()
def dataset(
    model: str = typer.Argument(..., help="Model to simulate [cphm, cmm, tdcm, thmm, aft_ln]"),
    n: int = typer.Option(100, help="Number of samples"),
    output: Optional[str] = typer.Option(None, "-o", help="Output CSV file. Prints to stdout if omitted."),
):
    """Generate survival data and optionally save to CSV."""
    df = generate(model=model, n=n)
    if output:
        df.to_csv(output, index=False)
        typer.echo(f"Saved dataset to {output}")
    else:
        typer.echo(df.to_csv(index=False))

if __name__ == "__main__":
    app()
