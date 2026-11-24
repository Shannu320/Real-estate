"""Model training module for regression models."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    MODELS_DIR, RANDOM_STATE, CV_FOLDS, PARAM_GRIDS, TARGET
)


def get_models() -> Dict[str, Any]:
    """Get dictionary of base models."""
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(random_state=RANDOM_STATE),
        "lasso": Lasso(random_state=RANDOM_STATE, max_iter=10000),
        "elastic_net": ElasticNet(random_state=RANDOM_STATE, max_iter=10000),
        "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "xgboost": xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "lightgbm": lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring: str = "neg_root_mean_squared_error"
) -> Dict[str, float]:
    """
    Perform cross-validation.

    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Number of folds
        scoring: Scoring metric

    Returns:
        Dictionary with CV results
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    rmse_scores = -scores  # Convert from negative

    return {
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "cv_rmse_scores": rmse_scores.tolist()
    }


def tune_hyperparameters(
    model: Any,
    param_grid: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    search_type: str = "random",
    n_iter: int = 50,
    scoring: str = "neg_root_mean_squared_error"
) -> Tuple[Any, Dict]:
    """
    Tune hyperparameters using GridSearch or RandomizedSearch.

    Args:
        model: Base model
        param_grid: Parameter grid
        X: Features
        y: Target
        cv: Cross-validation folds
        search_type: 'grid' or 'random'
        n_iter: Iterations for random search
        scoring: Scoring metric

    Returns:
        Best estimator and best parameters
    """
    if search_type == "grid":
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
    else:
        search = RandomizedSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_iter=n_iter, n_jobs=-1, verbose=1,
            random_state=RANDOM_STATE
        )

    search.fit(X, y)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score (RMSE): {-search.best_score_:,.2f}")

    return search.best_estimator_, search.best_params_


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune: bool = True,
    search_type: str = "random"
) -> Tuple[Any, Dict, Dict]:
    """
    Train a single model with optional hyperparameter tuning.

    Args:
        model_name: Name of model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        tune: Whether to tune hyperparameters
        search_type: 'grid' or 'random'

    Returns:
        Trained model, train metrics, test metrics
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]

    # Cross-validation on base model
    print("\nBase model cross-validation...")
    cv_results = cross_validate_model(model, X_train, y_train)
    print(f"CV RMSE: {cv_results['cv_rmse_mean']:,.2f} (+/- {cv_results['cv_rmse_std']:,.2f})")

    # Hyperparameter tuning
    if tune and model_name in PARAM_GRIDS:
        print(f"\nHyperparameter tuning ({search_type} search)...")
        model, best_params = tune_hyperparameters(
            model, PARAM_GRIDS[model_name], X_train, y_train,
            search_type=search_type
        )
    else:
        best_params = {}

    # Train final model
    print("\nTraining final model...")
    model.fit(X_train, y_train)

    # Evaluate on train and test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)

    print(f"\nTrain metrics: {train_metrics}")
    print(f"Test metrics: {test_metrics}")

    return model, train_metrics, test_metrics, cv_results, best_params


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_to_train: Optional[list] = None,
    tune: bool = True,
    search_type: str = "random"
) -> Dict[str, Dict]:
    """
    Train multiple models and compare.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        models_to_train: List of model names. If None, train all.
        tune: Whether to tune hyperparameters
        search_type: 'grid' or 'random'

    Returns:
        Dictionary with results for each model
    """
    if models_to_train is None:
        models_to_train = [
            "linear_regression", "ridge", "lasso",
            "decision_tree", "random_forest",
            "xgboost", "lightgbm"
        ]

    results = {}

    for model_name in models_to_train:
        try:
            model, train_metrics, test_metrics, cv_results, best_params = train_model(
                model_name, X_train, y_train, X_test, y_test,
                tune=tune, search_type=search_type
            )

            results[model_name] = {
                "model": model,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "cv_results": cv_results,
                "best_params": best_params
            }

            # Save model
            model_path = MODELS_DIR / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"Model saved to: {model_path}")

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results


def get_feature_importance(model: Any, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({"feature": feature_names})

    if hasattr(model, "coef_"):
        # Linear models
        importance_df["importance"] = np.abs(model.coef_)
        importance_df["type"] = "coefficient"
    elif hasattr(model, "feature_importances_"):
        # Tree-based models
        importance_df["importance"] = model.feature_importances_
        importance_df["type"] = "feature_importance"
    else:
        importance_df["importance"] = 0
        importance_df["type"] = "unknown"

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Calculate permutation importance.

    Args:
        model: Trained model
        X: Features
        y: Target
        n_repeats: Number of permutations
        random_state: Random seed

    Returns:
        DataFrame with permutation importance
    """
    from sklearn.inspection import permutation_importance as pi

    result = pi(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


def save_training_results(results: Dict, filepath: Path = MODELS_DIR / "training_results.pkl") -> Path:
    """Save training results (excluding model objects)."""
    # Create serializable copy
    serializable = {}
    for name, result in results.items():
        serializable[name] = {
            "train_metrics": result.get("train_metrics", {}),
            "test_metrics": result.get("test_metrics", {}),
            "cv_results": result.get("cv_results", {}),
            "best_params": result.get("best_params", {}),
            "error": result.get("error", None)
        }

    joblib.dump(serializable, filepath)
    print(f"Training results saved to: {filepath}")
    return filepath


def load_training_results(filepath: Path = MODELS_DIR / "training_results.pkl") -> Dict:
    """Load training results."""
    return joblib.load(filepath)


def select_best_model(results: Dict, metric: str = "test_RMSE") -> Tuple[str, Any]:
    """
    Select best model based on metric.

    Args:
        results: Training results dictionary
        metric: Metric to optimize (test_RMSE, test_MAE, test_R2, etc.)

    Returns:
        Best model name and model object
    """
    best_score = float("inf") if "RMSE" in metric or "MAE" in metric or "MSE" in metric else -float("inf")
    best_name = None
    best_model = None

    for name, result in results.items():
        if "error" in result:
            continue

        # Extract metric value
        if "test_metrics" in result:
            # Handle different metric key formats
            metric_key = metric.replace("test_", "")
            if metric_key in result["test_metrics"]:
                score = result["test_metrics"][metric_key]
            else:
                continue
        else:
            continue

        if "RMSE" in metric or "MAE" in metric or "MSE" in metric:
            if score < best_score:
                best_score = score
                best_name = name
                best_model = result["model"]
        else:
            if score > best_score:
                best_score = score
                best_name = name
                best_model = result["model"]

    print(f"\nBest model: {best_name} ({metric} = {best_score:.4f})")
    return best_name, best_model


if __name__ == "__main__":
    from src.data.preprocessing import load_processed_data

    X_train, X_test, y_train, y_test, scaler = load_processed_data()

    results = train_all_models(X_train, y_train, X_test, y_test)
    save_training_results(results)

    best_name, best_model = select_best_model(results, "test_RMSE")

    # Save best model separately
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    print(f"\nBest model ({best_name}) saved as best_model.pkl")