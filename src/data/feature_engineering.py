"""Feature engineering module for creating derived features and statistical analysis."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

from src.config import (
    DATA_PROCESSED, AGE_BANDS, CONDITION_THRESHOLD, GRADE_THRESHOLD,
    SEATTLE_CENTER, ENGINEERED_FEATURES
)


def calculate_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price per square foot."""
    df = df.copy()
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    return df


def calculate_property_age(df: pd.DataFrame, reference_year: int = 2015) -> pd.DataFrame:
    """Calculate property age from year built."""
    df = df.copy()
    df["age"] = reference_year - df["yr_built"]
    # Ensure age is not negative
    df["age"] = df["age"].clip(lower=0)
    return df


def create_renovation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flag for renovated properties."""
    df = df.copy()
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
    return df


def create_age_bands(df: pd.DataFrame, bands: List[Tuple] = AGE_BANDS) -> pd.DataFrame:
    """Create age band categories."""
    df = df.copy()

    def assign_band(age):
        for low, high, label in bands:
            if low <= age < high:
                return label
        return bands[-1][2]  # Last band for overflow

    df["age_band"] = df["age"].apply(assign_band)
    return df


def create_amenity_count(df: pd.DataFrame) -> pd.DataFrame:
    """Create amenity count feature."""
    df = df.copy()

    amenities = []

    # Waterfront
    if "waterfront" in df.columns:
        amenities.append((df["waterfront"] == 1).astype(int))

    # View (any view > 0)
    if "view" in df.columns:
        amenities.append((df["view"] > 0).astype(int))

    # Good condition (>= threshold)
    if "condition" in df.columns:
        amenities.append((df["condition"] >= CONDITION_THRESHOLD).astype(int))

    # High grade (>= threshold)
    if "grade" in df.columns:
        amenities.append((df["grade"] >= GRADE_THRESHOLD).astype(int))

    if amenities:
        df["amenity_count"] = sum(amenities)
    else:
        df["amenity_count"] = 0

    return df


def create_basement_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flag for properties with basement."""
    df = df.copy()
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
    return df


def create_living_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Create living area to lot area ratio."""
    df = df.copy()
    df["living_ratio"] = df["sqft_living"] / df["sqft_lot"]
    # Cap at reasonable values
    df["living_ratio"] = df["living_ratio"].clip(upper=1.0)
    return df


def create_zipcode_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Create zipcode frequency encoding."""
    df = df.copy()
    zipcode_counts = df["zipcode"].value_counts()
    df["zipcode_freq"] = df["zipcode"].map(zipcode_counts)
    return df


def calculate_distance_to_seattle(df: pd.DataFrame, center: Tuple[float, float] = SEATTLE_CENTER) -> pd.DataFrame:
    """Calculate distance to Seattle city center (approximate)."""
    df = df.copy()

    # Haversine formula for distance
    lat1, lon1 = np.radians(center)
    lat2 = np.radians(df["lat"])
    lon2 = np.radians(df["long"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Earth radius in miles
    r = 3959
    df["dist_to_seattle_center"] = c * r

    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Args:
        df: Input DataFrame with raw features

    Returns:
        DataFrame with engineered features
    """
    print("=== Feature Engineering ===")

    df = calculate_price_per_sqft(df)
    print("✓ Price per sqft")

    df = calculate_property_age(df)
    print("✓ Property age")

    df = create_renovation_flag(df)
    print("✓ Renovation flag")

    df = create_age_bands(df)
    print("✓ Age bands")

    df = create_amenity_count(df)
    print("✓ Amenity count")

    df = create_basement_flag(df)
    print("✓ Basement flag")

    df = create_living_ratio(df)
    print("✓ Living ratio")

    df = create_zipcode_frequency(df)
    print("✓ Zipcode frequency")

    df = calculate_distance_to_seattle(df)
    print("✓ Distance to Seattle center")

    print(f"\nAdded {len(ENGINEERED_FEATURES)} engineered features")
    return df


def statistical_analysis(df: pd.DataFrame, target: str = "price") -> Dict:
    """
    Perform statistical analysis on features.

    Args:
        df: Input DataFrame
        target: Target variable name

    Returns:
        Dictionary with statistical results
    """
    print("=== Statistical Analysis ===")

    results = {}

    # 1. Central tendency and dispersion for target
    target_data = df[target]
    results["target_stats"] = {
        "mean": target_data.mean(),
        "median": target_data.median(),
        "mode": target_data.mode().iloc[0] if not target_data.mode().empty else None,
        "std": target_data.std(),
        "var": target_data.var(),
        "skew": target_data.skew(),
        "kurtosis": target_data.kurtosis(),
        "iqr": target_data.quantile(0.75) - target_data.quantile(0.25),
        "cv": target_data.std() / target_data.mean()  # Coefficient of variation
    }

    # 2. Correlation with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target].sort_values(ascending=False)
    results["correlations"] = correlations.to_dict()

    # 3. VIF for multicollinearity
    # Select subset of numerical features for VIF (avoid too many)
    vif_features = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "sqft_above", "sqft_basement", "age", "lat", "long",
        "sqft_living15", "sqft_lot15", "amenity_count", "living_ratio",
        "zipcode_freq", "dist_to_seattle_center"
    ]
    vif_features = [f for f in vif_features if f in df.columns]

    vif_data = df[vif_features].dropna()
    if len(vif_data) > len(vif_features):
        vif_results = pd.DataFrame()
        vif_results["feature"] = vif_features
        vif_results["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(len(vif_features))]
        results["vif"] = vif_results.set_index("feature")["VIF"].to_dict()

        # Flag high VIF (> 5 or > 10)
        high_vif = vif_results[vif_results["VIF"] > 5]
        if len(high_vif) > 0:
            print(f"⚠ High VIF features (>5): {high_vif['feature'].tolist()}")

    # 4. Normality tests for target
    _, p_shapiro = stats.shapiro(target_data.sample(min(5000, len(target_data)), random_state=42))
    _, p_jarque = stats.jarque_bera(target_data)
    results["normality_tests"] = {
        "shapiro_p": p_shapiro,
        "jarque_bera_p": p_jarque,
        "is_normal_shapiro": p_shapiro > 0.05,
        "is_normal_jarque": p_jarque > 0.05
    }

    # 5. Price per sqft by categories
    if "price_per_sqft" in df.columns:
        results["price_per_sqft_by_zipcode"] = df.groupby("zipcode")["price_per_sqft"].agg(["mean", "median", "count"]).to_dict()
        results["price_per_sqft_by_age_band"] = df.groupby("age_band")["price_per_sqft"].agg(["mean", "median", "count"]).to_dict()
        results["price_per_sqft_by_condition"] = df.groupby("condition")["price_per_sqft"].agg(["mean", "median", "count"]).to_dict()
        results["price_per_sqft_by_grade"] = df.groupby("grade")["price_per_sqft"].agg(["mean", "median", "count"]).to_dict()

    print("✓ Central tendency & dispersion")
    print("✓ Correlations")
    print("✓ VIF analysis")
    print("✓ Normality tests")
    print("✓ Category breakdowns")

    return results


def save_engineered_data(df: pd.DataFrame, filepath: Path = DATA_PROCESSED / "kc_house_engineered.csv") -> Path:
    """Save engineered dataset."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Engineered data saved to: {filepath}")
    return filepath


def save_statistical_results(results: Dict, filepath: Path = DATA_PROCESSED / "statistical_analysis.json") -> Path:
    """Save statistical analysis results."""
    import json

    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Statistical results saved to: {filepath}")
    return filepath


def full_feature_engineering_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full feature engineering pipeline.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Engineered DataFrame and statistical results
    """
    print("=== Starting Feature Engineering Pipeline ===")

    # Create all features
    df = create_all_features(df)

    # Statistical analysis
    stats_results = statistical_analysis(df)

    # Save engineered data
    save_engineered_data(df)

    # Save statistical results
    save_statistical_results(stats_results)

    print("\n=== Feature Engineering Complete ===")
    print(f"Final shape: {df.shape}")
    print(f"Features: {list(df.columns)}")

    return df, stats_results


if __name__ == "__main__":
    from src.data.ingestion import load_raw_data
    from src.data.preprocessing import full_preprocessing_pipeline

    df = load_raw_data()
    X_train, X_test, y_train, y_test, scaler = full_preprocessing_pipeline(df)

    # Combine for feature engineering
    train_df = X_train.copy()
    train_df["price"] = y_train.values

    engineered_df, stats_results = full_feature_engineering_pipeline(train_df)