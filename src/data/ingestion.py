"""Data ingestion module for loading and validating property datasets."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import kagglehub
import shutil

from src.config import (
    DATA_RAW, DATASET_NAME, DATASET_URL, RANDOM_STATE
)


def download_kaggle_dataset(dataset_path: str = "harlfoxem/housesalesprediction") -> Path:
    """
    Download the King County House Sales dataset from Kaggle.

    Args:
        dataset_path: Kaggle dataset identifier

    Returns:
        Path to the downloaded CSV file
    """
    print(f"Downloading dataset from Kaggle: {dataset_path}")
    path = kagglehub.dataset_download(dataset_path)

    # Find the CSV file in the downloaded directory
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded dataset")

    source_file = csv_files[0]
    destination = DATA_RAW / DATASET_NAME

    # Copy to our data/raw directory
    shutil.copy2(source_file, destination)
    print(f"Dataset saved to: {destination}")

    return destination


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw property dataset.

    Args:
        filepath: Path to CSV file. If None, uses default location.

    Returns:
        DataFrame with raw property data
    """
    if filepath is None:
        filepath = DATA_RAW / DATASET_NAME

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}. Run download_kaggle_dataset() first.")

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Perform basic validation on the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with validation results
    """
    validation = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_ids": df.duplicated(subset=["id"]).sum() if "id" in df.columns else None,
        "price_stats": df["price"].describe().to_dict() if "price" in df.columns else None,
        "date_range": None
    }

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        validation["date_range"] = {
            "min": df["date"].min(),
            "max": df["date"].max()
        }

    return validation


def save_raw_data(df: pd.DataFrame, filepath: Optional[Path] = None) -> Path:
    """
    Save raw data to CSV.

    Args:
        df: DataFrame to save
        filepath: Output path. If None, uses default location.

    Returns:
        Path to saved file
    """
    if filepath is None:
        filepath = DATA_RAW / DATASET_NAME

    df.to_csv(filepath, index=False)
    print(f"Raw data saved to: {filepath}")
    return filepath


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive summary of the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for col in df.columns:
        col_data = df[col]
        summary_data.append({
            "column": col,
            "dtype": str(col_data.dtype),
            "non_null_count": col_data.notna().sum(),
            "null_count": col_data.isna().sum(),
            "null_pct": round(col_data.isna().mean() * 100, 2),
            "unique_count": col_data.nunique(),
        })

        if np.issubdtype(col_data.dtype, np.number):
            summary_data[-1].update({
                "mean": round(col_data.mean(), 2),
                "std": round(col_data.std(), 2),
                "min": col_data.min(),
                "median": col_data.median(),
                "max": col_data.max(),
                "skew": round(col_data.skew(), 2),
                "kurtosis": round(col_data.kurtosis(), 2)
            })

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Download and validate dataset
    try:
        csv_path = download_kaggle_dataset()
        df = load_raw_data(csv_path)
        validation = validate_data(df)

        print("\n=== Data Validation Summary ===")
        print(f"Shape: {validation['shape']}")
        print(f"Duplicate rows: {validation['duplicate_rows']}")
        print(f"Duplicate IDs: {validation['duplicate_ids']}")
        print(f"\nMissing values:")
        for col, count in validation['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")

        if validation['price_stats']:
            print(f"\nPrice stats:")
            for stat, val in validation['price_stats'].items():
                print(f"  {stat}: {val:,.0f}" if isinstance(val, (int, float)) else f"  {stat}: {val}")

        # Save raw copy
        save_raw_data(df)

        # Generate summary
        summary = get_data_summary(df)
        summary_path = DATA_RAW / "data_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nData summary saved to: {summary_path}")

    except Exception as e:
        print(f"Error: {e}")
        print("Trying to load from local file...")
        df = load_raw_data()
        validation = validate_data(df)
        print("Loaded existing data successfully")