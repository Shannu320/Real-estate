"""Property price prediction module."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import argparse

from src.config import MODELS_DIR, DATA_PROCESSED, TARGET


def load_model(model_name: str = "best_model", models_dir: Path = MODELS_DIR) -> object:
    """
    Load a trained model.

    Args:
        model_name: Name of model to load (without .pkl extension)
        models_dir: Directory containing models

    Returns:
        Loaded model object
    """
    model_path = models_dir / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Loaded model: {model_name}")
    return model


def load_scaler(scaler_path: Path = DATA_PROCESSED / "scaler.pkl") -> object:
    """Load the fitted scaler."""
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    return joblib.load(scaler_path)


def get_model_scaler(model_name: str = "best_model") -> tuple:
    """
    Get model and a properly fitted scaler for the model's expected features.

    The original scaler was fit on different features than what the model expects.
    This creates a new scaler fitted on the model's expected features using training data.
    """
    model = load_model(model_name)

    # Get the model's expected features
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback to the 29 features we know the model expects
        feature_names = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
            'sqft_living15', 'sqft_lot15',
            'sale_year', 'sale_month', 'sale_quarter', 'sale_dayofweek',
            'age', 'is_renovated', 'amenity_count', 'has_basement', 'living_ratio',
            'zipcode_freq', 'dist_to_seattle_center'
        ]

    # Load RAW engineered training data to fit a new scaler
    train_eng = pd.read_csv(DATA_PROCESSED / 'kc_house_engineered_raw.csv')

    # Prepare features matching model's expected features
    X_train = train_eng[feature_names].copy()
    X_train = X_train.fillna(0)

    # Fit new scaler on these features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    return model, scaler, feature_names


def prepare_input_data(input_data: Union[Dict, pd.DataFrame], feature_names: List[str]) -> pd.DataFrame:
    """
    Prepare input data for prediction.

    Args:
        input_data: Dictionary or DataFrame with property features
        feature_names: Expected feature names in order

    Returns:
        Prepared DataFrame
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Ensure all required features are present
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Select and order features
    df = df[feature_names]

    return df


def predict_price(
    model: object,
    scaler: object,
    input_data: Union[Dict, pd.DataFrame],
    feature_names: List[str],
    return_interval: bool = True,
    interval_width: float = 0.95
) -> Dict:
    """
    Predict property price with optional prediction interval.

    Args:
        model: Trained model
        scaler: Fitted scaler
        input_data: Property features
        feature_names: Feature names in order
        return_interval: Whether to return prediction interval
        interval_width: Width of prediction interval (e.g., 0.95 for 95%)

    Returns:
        Dictionary with prediction and optionally interval
    """
    # Prepare input
    X = prepare_input_data(input_data, feature_names)

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    result = {
        "predicted_price": float(prediction),
        "predicted_price_formatted": f"${prediction:,.0f}"
    }

    if return_interval:
        # For prediction intervals, we need residual standard error
        # This would ideally come from training; using a default estimate
        # In practice, load this from saved evaluation results
        residual_se = 85000  # Approximate from typical model performance
        z_score = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(interval_width, 1.96)
        margin = z_score * residual_se

        lower = max(0, prediction - margin)
        upper = prediction + margin

        result["prediction_interval"] = {
            "lower": float(lower),
            "upper": float(upper),
            "width": interval_width,
            "lower_formatted": f"${lower:,.0f}",
            "upper_formatted": f"${upper:,.0f}"
        }

    return result


def batch_predict(
    model: object,
    scaler: object,
    input_df: pd.DataFrame,
    feature_names: List[str],
    return_interval: bool = True
) -> pd.DataFrame:
    """
    Predict prices for multiple properties.

    Args:
        model: Trained model
        scaler: Fitted scaler
        input_df: DataFrame with property features
        feature_names: Feature names in order
        return_interval: Whether to include prediction intervals

    Returns:
        DataFrame with predictions
    """
    X = input_df[feature_names]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    result_df = input_df.copy()
    result_df["predicted_price"] = predictions
    result_df["predicted_price_formatted"] = result_df["predicted_price"].apply(lambda x: f"${x:,.0f}")

    if return_interval:
        residual_se = 85000
        margin = 1.96 * residual_se
        result_df["pred_lower"] = (result_df["predicted_price"] - margin).clip(lower=0)
        result_df["pred_upper"] = result_df["predicted_price"] + margin
        result_df["pred_lower_formatted"] = result_df["pred_lower"].apply(lambda x: f"${x:,.0f}")
        result_df["pred_upper_formatted"] = result_df["pred_upper"].apply(lambda x: f"${x:,.0f}")

    return result_df


def predict_from_file(
    input_file: Path,
    model_name: str = "best_model",
    output_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Predict prices from CSV file.

    Args:
        input_file: Path to input CSV
        model_name: Model to use
        output_file: Path to save predictions (optional)

    Returns:
        DataFrame with predictions
    """
    # Load model and scaler
    model = load_model(model_name)
    scaler = load_scaler()

    # Load feature names from training data
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    feature_names = X_train.columns.tolist()

    # Load input data
    input_df = pd.read_csv(input_file)

    # Predict
    results = batch_predict(model, scaler, input_df, feature_names)

    if output_file:
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

    return results


def interactive_predict(model_name: str = "best_model") -> None:
    """Interactive prediction from command line input."""
    # Use get_model_scaler to get properly fitted scaler
    model, scaler, feature_names = get_model_scaler(model_name)

    print("=== Property Price Prediction ===")
    print("Enter property features (press Enter for defaults):")
    print()

    # Get user input for key features
    input_data = {}

    # Numerical features with defaults
    defaults = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000,
        "sqft_lot": 8000,
        "floors": 1.5,
        "sqft_above": 1800,
        "sqft_basement": 200,
        "yr_built": 1990,
        "yr_renovated": 0,
        "lat": 47.5,
        "long": -122.2,
        "sqft_living15": 1900,
        "sqft_lot15": 7500
    }

    for feature, default in defaults.items():
        val = input(f"{feature} [{default}]: ").strip()
        input_data[feature] = float(val) if val else default

    # Categorical features
    input_data["waterfront"] = int(input("waterfront (0/1) [0]: ").strip() or 0)
    input_data["view"] = int(input("view (0-4) [0]: ").strip() or 0)
    input_data["condition"] = int(input("condition (1-5) [3]: ").strip() or 3)
    input_data["grade"] = int(input("grade (1-13) [7]: ").strip() or 7)
    input_data["zipcode"] = int(input("zipcode [98103]: ").strip() or 98103)

    # Compute engineered features from user input
    from src.config import AGE_BANDS, CONDITION_THRESHOLD, GRADE_THRESHOLD, SEATTLE_CENTER
    import pandas as pd
    import numpy as np

    # Create a single-row dataframe to compute engineered features
    df = pd.DataFrame([input_data])

    # Property age
    df['age'] = 2015 - df['yr_built']
    df['age'] = df['age'].clip(lower=0)

    # Renovation flag
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)

    # Amenity count
    amenities = []
    amenities.append((df['waterfront'] == 1).astype(int))
    amenities.append((df['view'] > 0).astype(int))
    amenities.append((df['condition'] >= CONDITION_THRESHOLD).astype(int))
    amenities.append((df['grade'] >= GRADE_THRESHOLD).astype(int))
    df['amenity_count'] = sum(amenities)

    # Basement flag
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)

    # Living ratio
    df['living_ratio'] = df['sqft_living'] / df['sqft_lot']
    df['living_ratio'] = df['living_ratio'].clip(upper=1.0)

    # Zipcode frequency - load from training data (raw engineered)
    train_eng = pd.read_csv(DATA_PROCESSED / 'kc_house_engineered_raw.csv')
    zipcode_counts = train_eng['zipcode'].value_counts()
    df['zipcode_freq'] = df['zipcode'].map(zipcode_counts).fillna(1)

    # Distance to Seattle center
    lat1, lon1 = SEATTLE_CENTER
    lat2 = df['lat'].values[0]
    lon2 = df['long'].values[0]
    from math import radians, sin, cos, sqrt, asin
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    df['dist_to_seattle_center'] = c * 3959  # miles

    # Date features (use median from training)
    df['sale_year'] = 2014
    df['sale_month'] = 6
    df['sale_quarter'] = 2
    df['sale_dayofweek'] = 2

    # Update input_data with engineered features
    for col in df.columns:
        input_data[col] = df[col].values[0]

    # Predict
    result = predict_price(model, scaler, input_data, feature_names)

    print(f"\n{'='*50}")
    print(f"Predicted Price: {result['predicted_price_formatted']}")
    if "prediction_interval" in result:
        pi = result["prediction_interval"]
        print(f"{int(pi['width']*100)}% Prediction Interval: {pi['lower_formatted']} - {pi['upper_formatted']}")
    print(f"{'='*50}")


def main():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(description="Predict property prices")
    parser.add_argument("--model", default="best_model", help="Model name")
    parser.add_argument("--input", help="Input CSV file")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_predict(args.model)
    elif args.input:
        predict_from_file(Path(args.input), args.model, Path(args.output) if args.output else None)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()