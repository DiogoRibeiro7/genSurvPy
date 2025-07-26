"""
Utilities for summarizing and validating survival datasets.

This module provides functions to summarize survival data,
check data quality, and identify potential issues.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


def summarize_survival_dataset(
    data: pd.DataFrame,
    time_col: str = "time",
    status_col: str = "status",
    id_col: Optional[str] = None,
    covariate_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of a survival dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing survival data.
    time_col : str, default="time"
        Name of the column containing time-to-event values.
    status_col : str, default="status"
        Name of the column containing event indicators (1=event, 0=censored).
    id_col : str, optional
        Name of the column containing subject identifiers.
    covariate_cols : list of str, optional
        List of column names to include as covariates in the summary.
        If None, all columns except time_col, status_col, and id_col are considered.
    verbose : bool, default=True
        Whether to print the summary to console.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all summary statistics.
    
    Examples
    --------
    >>> from gen_surv import generate
    >>> from gen_surv.summary import summarize_survival_dataset
    >>> 
    >>> # Generate example data
    >>> df = generate(model="cphm", n=100, model_cens="uniform",
    ...               cens_par=1.0, beta=0.5, covariate_range=2.0)
    >>> 
    >>> # Summarize the dataset
    >>> summary = summarize_survival_dataset(df)
    """
    # Validate input columns
    for col in [time_col, status_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    
    if id_col is not None and id_col not in data.columns:
        raise ValueError(f"ID column '{id_col}' not found in data")
    
    # Determine covariate columns
    if covariate_cols is None:
        exclude_cols = {time_col, status_col}
        if id_col is not None:
            exclude_cols.add(id_col)
        covariate_cols = [col for col in data.columns if col not in exclude_cols]
    else:
        missing_cols = [col for col in covariate_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Covariate columns not found in data: {missing_cols}")
    
    # Basic dataset information
    n_subjects = len(data)
    if id_col is not None:
        n_unique_ids = data[id_col].nunique()
    else:
        n_unique_ids = n_subjects
    
    # Event information
    n_events = data[status_col].sum()
    n_censored = n_subjects - n_events
    event_rate = n_events / n_subjects
    
    # Time statistics
    time_min = data[time_col].min()
    time_max = data[time_col].max()
    time_mean = data[time_col].mean()
    time_median = data[time_col].median()
    
    # Data quality checks
    n_missing_time = data[time_col].isna().sum()
    n_missing_status = data[status_col].isna().sum()
    n_negative_time = (data[time_col] < 0).sum()
    n_invalid_status = data[~data[status_col].isin([0, 1])].shape[0]
    
    # Covariate summaries
    covariate_stats = {}
    for col in covariate_cols:
        col_data = data[col]
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numeric:
            covariate_stats[col] = {
                "type": "numeric",
                "min": col_data.min(),
                "max": col_data.max(),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "missing": col_data.isna().sum(),
                "unique_values": col_data.nunique()
            }
        else:
            # Categorical/string
            covariate_stats[col] = {
                "type": "categorical",
                "n_categories": col_data.nunique(),
                "top_categories": col_data.value_counts().head(5).to_dict(),
                "missing": col_data.isna().sum()
            }
    
    # Compile the summary
    summary = {
        "dataset_info": {
            "n_subjects": n_subjects,
            "n_unique_ids": n_unique_ids,
            "n_covariates": len(covariate_cols)
        },
        "event_info": {
            "n_events": n_events,
            "n_censored": n_censored,
            "event_rate": event_rate
        },
        "time_info": {
            "min": time_min,
            "max": time_max,
            "mean": time_mean,
            "median": time_median
        },
        "data_quality": {
            "missing_time": n_missing_time,
            "missing_status": n_missing_status,
            "negative_time": n_negative_time,
            "invalid_status": n_invalid_status,
            "overall_quality": "good" if (n_missing_time + n_missing_status + n_negative_time + n_invalid_status) == 0 else "issues_detected"
        },
        "covariates": covariate_stats
    }
    
    # Print summary if requested
    if verbose:
        _print_summary(summary, time_col, status_col, id_col, covariate_cols)
    
    return summary


def check_survival_data_quality(
    data: pd.DataFrame,
    time_col: str = "time",
    status_col: str = "status",
    id_col: Optional[str] = None,
    min_time: float = 0.0,
    max_time: Optional[float] = None,
    status_values: Optional[List[int]] = None,
    fix_issues: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Check for common issues in survival data and optionally fix them.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing survival data.
    time_col : str, default="time"
        Name of the column containing time-to-event values.
    status_col : str, default="status"
        Name of the column containing event indicators.
    id_col : str, optional
        Name of the column containing subject identifiers.
    min_time : float, default=0.0
        Minimum acceptable value for time column.
    max_time : float, optional
        Maximum acceptable value for time column.
    status_values : list of int, optional
        List of valid status values. Default is [0, 1].
    fix_issues : bool, default=False
        Whether to attempt fixing issues (returns a modified DataFrame).
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Tuple containing (possibly fixed) DataFrame and issues report.
    
    Examples
    --------
    >>> from gen_surv import generate
    >>> from gen_surv.summary import check_survival_data_quality
    >>> 
    >>> # Generate example data with some issues
    >>> df = generate(model="cphm", n=100, model_cens="uniform",
    ...               cens_par=1.0, beta=0.5, covariate_range=2.0)
    >>> # Introduce some issues
    >>> df.loc[0, "time"] = np.nan
    >>> df.loc[1, "status"] = 2  # Invalid status
    >>> 
    >>> # Check and fix issues
    >>> fixed_df, issues = check_survival_data_quality(df, fix_issues=True)
    >>> print(issues)
    """
    if status_values is None:
        status_values = [0, 1]
    
    # Make a copy to avoid modifying the original
    if fix_issues:
        data = data.copy()
    
    # Initialize issues report
    issues = {
        "missing_data": {
            "time": 0,
            "status": 0,
            "id": 0 if id_col else None
        },
        "invalid_values": {
            "negative_time": 0,
            "excessive_time": 0,
            "invalid_status": 0
        },
        "duplicates": {
            "duplicate_rows": 0,
            "duplicate_ids": 0 if id_col else None
        },
        "modifications": {
            "rows_dropped": 0,
            "values_fixed": 0
        }
    }
    
    # Check for missing values
    issues["missing_data"]["time"] = data[time_col].isna().sum()
    issues["missing_data"]["status"] = data[status_col].isna().sum()
    if id_col:
        issues["missing_data"]["id"] = data[id_col].isna().sum()
    
    # Check for invalid values
    issues["invalid_values"]["negative_time"] = (data[time_col] < min_time).sum()
    if max_time is not None:
        issues["invalid_values"]["excessive_time"] = (data[time_col] > max_time).sum()
    issues["invalid_values"]["invalid_status"] = data[~data[status_col].isin(status_values)].shape[0]
    
    # Check for duplicates
    issues["duplicates"]["duplicate_rows"] = data.duplicated().sum()
    if id_col:
        issues["duplicates"]["duplicate_ids"] = data[id_col].duplicated().sum()
    
    # Fix issues if requested
    if fix_issues:
        original_rows = len(data)
        modified_values = 0
        
        # Handle missing values
        data = data.dropna(subset=[time_col, status_col])
        
        # Handle invalid values
        if min_time > 0:
            # Set negative or too small times to min_time
            mask = data[time_col] < min_time
            if mask.any():
                data.loc[mask, time_col] = min_time
                modified_values += mask.sum()
        
        if max_time is not None:
            # Cap excessively large times
            mask = data[time_col] > max_time
            if mask.any():
                data.loc[mask, time_col] = max_time
                modified_values += mask.sum()
        
        # Fix invalid status values
        mask = ~data[status_col].isin(status_values)
        if mask.any():
            # Default to censored (0) for invalid status
            data.loc[mask, status_col] = 0
            modified_values += mask.sum()
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Update modification counts
        issues["modifications"]["rows_dropped"] = original_rows - len(data)
        issues["modifications"]["values_fixed"] = modified_values
    
    return data, issues


def _print_summary(
    summary: Dict[str, Any],
    time_col: str,
    status_col: str,
    id_col: Optional[str],
    covariate_cols: List[str]
) -> None:
    """
    Print a formatted summary of survival data.
    
    Parameters
    ----------
    summary : Dict[str, Any]
        Summary dictionary from summarize_survival_dataset.
    time_col : str
        Name of the time column.
    status_col : str
        Name of the status column.
    id_col : str, optional
        Name of the ID column.
    covariate_cols : List[str]
        List of covariate column names.
    """
    print("=" * 60)
    print("SURVIVAL DATASET SUMMARY")
    print("=" * 60)
    
    # Dataset info
    print("\nDATASET INFORMATION:")
    print(f"  Subjects:     {summary['dataset_info']['n_subjects']}")
    if id_col:
        print(f"  Unique IDs:   {summary['dataset_info']['n_unique_ids']}")
    print(f"  Covariates:   {summary['dataset_info']['n_covariates']}")
    
    # Event info
    print("\nEVENT INFORMATION:")
    print(f"  Events:       {summary['event_info']['n_events']} " +
          f"({summary['event_info']['event_rate']:.1%})")
    print(f"  Censored:     {summary['event_info']['n_censored']} " +
          f"({1 - summary['event_info']['event_rate']:.1%})")
    
    # Time info
    print(f"\nTIME VARIABLE ({time_col}):")
    print(f"  Range:        {summary['time_info']['min']:.2f} to {summary['time_info']['max']:.2f}")
    print(f"  Mean:         {summary['time_info']['mean']:.2f}")
    print(f"  Median:       {summary['time_info']['median']:.2f}")
    
    # Data quality
    print("\nDATA QUALITY:")
    quality_issues = (
        summary['data_quality']['missing_time'] +
        summary['data_quality']['missing_status'] +
        summary['data_quality']['negative_time'] +
        summary['data_quality']['invalid_status']
    )
    
    if quality_issues == 0:
        print("  ✓ No issues detected")
    else:
        print("  ✗ Issues detected:")
        if summary['data_quality']['missing_time'] > 0:
            print(f"    - Missing time values: {summary['data_quality']['missing_time']}")
        if summary['data_quality']['missing_status'] > 0:
            print(f"    - Missing status values: {summary['data_quality']['missing_status']}")
        if summary['data_quality']['negative_time'] > 0:
            print(f"    - Negative time values: {summary['data_quality']['negative_time']}")
        if summary['data_quality']['invalid_status'] > 0:
            print(f"    - Invalid status values: {summary['data_quality']['invalid_status']}")
    
    # Covariates
    print("\nCOVARIATES:")
    if not covariate_cols:
        print("  No covariates found")
    else:
        for col, stats in summary['covariates'].items():
            print(f"  {col}:")
            if stats['type'] == 'numeric':
                print("    Type:         Numeric")
                print(f"    Range:        {stats['min']:.2f} to {stats['max']:.2f}")
                print(f"    Mean:         {stats['mean']:.2f}")
                print(f"    Missing:      {stats['missing']}")
            else:
                print("    Type:         Categorical")
                print(f"    Categories:   {stats['n_categories']}")
                print(f"    Missing:      {stats['missing']}")
    
    print("\n" + "=" * 60)


def compare_survival_datasets(
    datasets: Dict[str, pd.DataFrame],
    time_col: str = "time",
    status_col: str = "status",
    covariate_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple survival datasets and summarize their differences.
    
    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to DataFrames.
    time_col : str, default="time"
        Name of the time column in each dataset.
    status_col : str, default="status"
        Name of the status column in each dataset.
    covariate_cols : List[str], optional
        List of covariate columns to compare. If None, compares all common columns.
        
    Returns
    -------
    pd.DataFrame
        Comparison table with datasets as columns and metrics as rows.
    
    Examples
    --------
    >>> from gen_surv import generate
    >>> from gen_surv.summary import compare_survival_datasets
    >>> 
    >>> # Generate datasets with different parameters
    >>> datasets = {
    ...     "CPHM": generate(model="cphm", n=100, model_cens="uniform",
    ...                    cens_par=1.0, beta=0.5, covariate_range=2.0),
    ...     "Weibull AFT": generate(model="aft_weibull", n=100, beta=[0.5], 
    ...                           shape=1.5, scale=1.0, model_cens="uniform", cens_par=1.0)
    ... }
    >>> 
    >>> # Compare datasets
    >>> comparison = compare_survival_datasets(datasets)
    >>> print(comparison)
    """
    if not datasets:
        raise ValueError("No datasets provided for comparison")
    
    # Find common columns if covariate_cols not specified
    if covariate_cols is None:
        all_columns = [set(df.columns) for df in datasets.values()]
        common_columns = set.intersection(*all_columns)
        common_columns -= {time_col, status_col}  # Remove time and status
        covariate_cols = sorted(list(common_columns))
    
    # Calculate summaries for each dataset
    summaries = {}
    for name, data in datasets.items():
        summaries[name] = summarize_survival_dataset(
            data, time_col, status_col, 
            covariate_cols=covariate_cols, verbose=False
        )
    
    # Construct the comparison DataFrame
    comparison_data = {}
    
    # Dataset info
    comparison_data["n_subjects"] = {
        name: summary["dataset_info"]["n_subjects"] 
        for name, summary in summaries.items()
    }
    comparison_data["n_events"] = {
        name: summary["event_info"]["n_events"] 
        for name, summary in summaries.items()
    }
    comparison_data["event_rate"] = {
        name: summary["event_info"]["event_rate"] 
        for name, summary in summaries.items()
    }
    
    # Time info
    comparison_data["time_min"] = {
        name: summary["time_info"]["min"] 
        for name, summary in summaries.items()
    }
    comparison_data["time_max"] = {
        name: summary["time_info"]["max"] 
        for name, summary in summaries.items()
    }
    comparison_data["time_mean"] = {
        name: summary["time_info"]["mean"] 
        for name, summary in summaries.items()
    }
    comparison_data["time_median"] = {
        name: summary["time_info"]["median"] 
        for name, summary in summaries.items()
    }
    
    # Covariate info (means for numeric)
    for col in covariate_cols:
        for name, summary in summaries.items():
            if col in summary["covariates"]:
                col_stats = summary["covariates"][col]
                if col_stats["type"] == "numeric":
                    if f"{col}_mean" not in comparison_data:
                        comparison_data[f"{col}_mean"] = {}
                    comparison_data[f"{col}_mean"][name] = col_stats["mean"]
    
    # Create the DataFrame
    comparison_df = pd.DataFrame(comparison_data).T
    
    return comparison_df
