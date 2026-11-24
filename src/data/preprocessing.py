"""Data preprocessing module for cleaning and preparing property data."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

from src.config import (
    DATA_PROCESSED, TARGET, DROP_FEATURES, CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES, RANDOM_STATE, TEST_SIZE
)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # yr_renovated: 0 means not renovated, fill NaN with 0
    if "yr_renovated" in df.columns:
        df["yr_renovated"] = df["yr_renovated"].fillna(0)

    # For any other numerical columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates. Default: all columns

    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()
    initial_shape = df.shape[0]

    if subset is None:
        df = df.drop_duplicates()
    else:
        df = df.drop_duplicates(subset=subset)

    removed = initial_shape - df.shape[0]
    if removed > 0:
        print(f"Removed {removed} duplicate rows")

    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using IQR method.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        factor: IQR multiplier (default 1.5)

    Returns:
        DataFrame with outlier flags
    """
    df = df.copy()
    outlier_flags = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outlier_flags[f"{col}_outlier"] = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_flags[f"{col}_lower_bound"] = lower_bound
        outlier_flags[f"{col}_upper_bound"] = upper_bound

        n_outliers = outlier_flags[f"{col}_outlier"].sum()
        print(f"{col}: {n_outliers} outliers detected (bounds: {lower_bound:.2f}, {upper_bound:.2f})")

    return outlier_flags


def remove_outliers(df: pd.DataFrame, outlier_flags: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Remove rows with outliers in specified columns.

    Args:
        df: Input DataFrame
        outlier_flags: DataFrame with outlier flags from detect_outliers_iqr
        columns: Columns to use for outlier removal

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    initial_shape = df.shape[0]

    # Combine outlier flags - remove if ANY specified column has outlier
    combined_mask = pd.Series(False, index=df.index)
    for col in columns:
        flag_col = f"{col}_outlier"
        if flag_col in outlier_flags.columns:
            combined_mask = combined_mask | outlier_flags[flag_col]

    df = df[~combined_mask].reset_index(drop=True)
    removed = initial_shape - df.shape[0]
    print(f"Removed {removed} rows with outliers ({removed/initial_shape*100:.2f}% of data)")

    return df


def convert_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date column to useful features.

    Args:
        df: Input DataFrame with 'date' column

    Returns:
        DataFrame with date features extracted
    """
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["sale_year"] = df["date"].dt.year
        df["sale_month"] = df["date"].dt.month
        df["sale_quarter"] = df["date"].dt.quarter
        df["sale_dayofweek"] = df["date"].dt.dayofweek

    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "standard",
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Scale numerical features.

    Args:
        X_train: Training features
        X_test: Test features
        scaler_type: 'standard' or 'robust'
        columns: Columns to scale. If None, scale all numerical columns.

    Returns:
        Scaled X_train, X_test, and fitted scaler
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if columns is None:
        columns = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])

    return X_train, X_test, scaler


def prepare_features_target(
    df: pd.DataFrame,
    target: str = TARGET,
    drop_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.

    Args:
        df: Input DataFrame
        target: Target column name
        drop_features: Features to drop

    Returns:
        X (features), y (target)
    """
    if drop_features is None:
        drop_features = DROP_FEATURES

    # Ensure target is not in drop_features
    drop_features = [f for f in drop_features if f != target]

    X = df.drop(columns=drop_features, errors="ignore")
    y = df[target]

    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified split based on target quantiles.

    Args:
        X: Features
        y: Target
        test_size: Proportion for test set
        random_state: Random seed
        n_bins: Number of quantile bins for stratification

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Create stratification bins based on target quantiles
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_binned
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: object,
    filepath: Path = DATA_PROCESSED
) -> None:
    """
    Save processed datasets and scaler.

    Args:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Target Series
        scaler: Fitted scaler object
        filepath: Output directory
    """
    filepath.mkdir(parents=True, exist_ok=True)

    # Save features and targets
    X_train.to_csv(filepath / "X_train.csv", index=False)
    X_test.to_csv(filepath / "X_test.csv", index=False)
    y_train.to_csv(filepath / "y_train.csv", index=False, header=True)
    y_test.to_csv(filepath / "y_test.csv", index=False, header=True)

    # Save scaler
    joblib.dump(scaler, filepath / "scaler.pkl")

    print(f"Processed data saved to: {filepath}")


def load_processed_data(filepath: Path = DATA_PROCESSED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Load processed datasets and scaler.

    Args:
        filepath: Input directory

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    X_train = pd.read_csv(filepath / "X_train.csv")
    X_test = pd.read_csv(filepath / "X_test.csv")
    y_train = pd.read_csv(filepath / "y_train.csv").squeeze()
    y_test = pd.read_csv(filepath / "y_test.csv").squeeze()
    scaler = joblib.load(filepath / "scaler.pkl")

    return X_train, X_test, y_train, y_test, scaler


def full_preprocessing_pipeline(
    df: pd.DataFrame,
    outlier_columns: Optional[List[str]] = None,
    scaler_type: str = "standard"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Run the full preprocessing pipeline.

    Args:
        df: Raw DataFrame
        outlier_columns: Columns to check for outliers
        scaler_type: Type of scaler to use

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    print("=== Starting Preprocessing Pipeline ===")

    # 1. Handle missing values
    print("\n1. Handling missing values...")
    df = handle_missing_values(df)

    # 2. Remove duplicates
    print("\n2. Removing duplicates...")
    df = remove_duplicates(df, subset=["id"])

    # 3. Convert date features
    print("\n3. Converting date features...")
    df = convert_date_features(df)

    # 4. Detect and remove outliers
    if outlier_columns is None:
        outlier_columns = ["price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]

    print("\n4. Detecting outliers...")
    outlier_flags = detect_outliers_iqr(df, outlier_columns)
    print("\n5. Removing outliers...")
    df = remove_outliers(df, outlier_flags, outlier_columns)

    # 5. Prepare features and target
    print("\n6. Preparing features and target...")
    X, y = prepare_features_target(df)

    # 6. Train/test split
    print("\n7. Splitting train/test...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)

    # 7. Scale features
    print("\n8. Scaling features...")
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train, X_test, scaler = scale_features(X_train, X_test, scaler_type, numerical_cols)

    # 8. Save processed data
    print("\n9. Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

    print("\n=== Preprocessing Complete ===")
    print(f"Final train shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    from src.data.ingestion import load_raw_data

    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler = full_preprocessing_pipeline(df)