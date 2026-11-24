"""Project configuration settings."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
POWERBI_DIR = PROJECT_ROOT / "powerbi"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset
DATASET_NAME = "kc_house_data.csv"
DATASET_URL = "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/download"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target variable
TARGET = "price"

# Features to drop (leakage or redundant)
DROP_FEATURES = ["id", "date", "price"]

# Categorical features
CATEGORICAL_FEATURES = ["zipcode", "waterfront", "view", "condition", "grade"]

# Numerical features
NUMERICAL_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
    "lat", "long", "sqft_living15", "sqft_lot15"
]

# Engineered features
ENGINEERED_FEATURES = [
    "price_per_sqft", "age", "is_renovated", "age_band",
    "amenity_count", "has_basement", "living_ratio",
    "zipcode_freq", "dist_to_seattle_center"
]

# Seattle center coordinates (approximate)
SEATTLE_CENTER = (47.6062, -122.3321)

# Model hyperparameter grids - LIGHTWEIGHT for resource efficiency
PARAM_GRIDS = {
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0]
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt"]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.9, 1.0]
    }
}

# Age bands
AGE_BANDS = [
    (0, 10, "0-10"),
    (10, 20, "10-20"),
    (20, 40, "20-40"),
    (40, 60, "40-60"),
    (60, float("inf"), "60+")
]

# Condition/Grade thresholds for amenity count
CONDITION_THRESHOLD = 4
GRADE_THRESHOLD = 8

# Ensure directories exist
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL, MODELS_DIR, REPORTS_DIR, POWERBI_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)