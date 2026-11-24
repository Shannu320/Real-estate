"""Model evaluation and comparison module."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE


def load_all_models(models_dir: Path = MODELS_DIR) -> Dict:
    """Load all saved models."""
    models = {}
    for model_file in models_dir.glob("*.pkl"):
        if model_file.stem not in ["scaler", "training_results", "best_model"]:
            models[model_file.stem] = joblib.load(model_file)
    return models


def evaluate_model_detailed(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model"
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        X: Features
        y: True target
        model_name: Name for reporting

    Returns:
        Dictionary with detailed evaluation results
    """
    y_pred = model.predict(X)

    # Basic metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    # Residuals
    residuals = y - y_pred

    # Residual statistics
    residual_stats = {
        "mean": residuals.mean(),
        "std": residuals.std(),
        "skew": stats.skew(residuals),
        "kurtosis": stats.kurtosis(residuals),
        "median": residuals.median(),
        "iqr": residuals.quantile(0.75) - residuals.quantile(0.25)
    }

    # Normality tests
    _, p_shapiro = stats.shapiro(residuals.sample(min(5000, len(residuals)), random_state=RANDOM_STATE))
    _, p_jarque = stats.jarque_bera(residuals)

    # Heteroscedasticity test (Breusch-Pagan)
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_lm, bp_p, _, _ = het_breuschpagan(residuals, X)
        bp_result = {"lm_stat": bp_lm, "p_value": bp_p, "heteroscedastic": bp_p < 0.05}
    except:
        bp_result = {"error": "Could not compute Breusch-Pagan test"}

    # Prediction intervals (using residual standard error)
    residual_se = residuals.std()
    pred_interval_95 = 1.96 * residual_se
    pred_interval_80 = 1.28 * residual_se

    # Error by price quantiles
    price_bins = pd.qcut(y, q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    error_by_bin = pd.DataFrame({"actual": y, "predicted": y_pred, "residual": residuals, "bin": price_bins})
    error_by_bin_summary = error_by_bin.groupby("bin").agg({
        "residual": ["mean", "std", "median"],
        "actual": "count"
    }).round(2)

    return {
        "model_name": model_name,
        "metrics": {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        },
        "residual_stats": residual_stats,
        "normality_tests": {
            "shapiro_p": p_shapiro,
            "jarque_bera_p": p_jarque,
            "is_normal_shapiro": p_shapiro > 0.05,
            "is_normal_jarque": p_jarque > 0.05
        },
        "heteroscedasticity": bp_result,
        "prediction_intervals": {
            "95%": pred_interval_95,
            "80%": pred_interval_80,
            "residual_se": residual_se
        },
        "error_by_price_bin": error_by_bin_summary.to_dict(),
        "residuals": residuals.values,
        "predictions": y_pred
    }


def compare_models(
    results: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path = MODELS_DIR
) -> pd.DataFrame:
    """
    Compare multiple models on test set.

    Args:
        results: Dictionary of training results
        X_test: Test features
        y_test: Test target
        models_dir: Directory containing saved models

    Returns:
        Comparison DataFrame
    """
    comparison = []

    for name, result in results.items():
        if "error" in result and result["error"] is not None:
            comparison.append({"model": name, "error": result["error"]})
            continue

        # Load model from disk
        model_path = models_dir / f"{name}.pkl"
        if not model_path.exists():
            comparison.append({"model": name, "error": "Model file not found"})
            continue

        model = joblib.load(model_path)
        eval_result = evaluate_model_detailed(model, X_test, y_test, name)

        row = {"model": name}
        row.update(eval_result["metrics"])
        row.update({
            "cv_rmse_mean": result.get("cv_results", {}).get("cv_rmse_mean", np.nan),
            "cv_rmse_std": result.get("cv_results", {}).get("cv_rmse_std", np.nan),
            "residual_mean": eval_result["residual_stats"]["mean"],
            "residual_std": eval_result["residual_stats"]["std"],
            "shapiro_p": eval_result["normality_tests"]["shapiro_p"],
            "bp_p": eval_result["heteroscedasticity"].get("p_value", np.nan)
        })
        comparison.append(row)

    df = pd.DataFrame(comparison)
    # Only sort rows that have RMSE (skip error rows)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE", na_position="last").reset_index(drop=True)
    return df


def plot_residuals(
    results: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 4,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot residuals for top models.

    Args:
        results: Training results
        X_test: Test features
        y_test: Test target
        top_n: Number of top models to plot
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    comparison = compare_models(results, X_test, y_test)
    top_models = comparison.head(top_n)["model"].tolist()

    fig, axes = plt.subplots(top_n, 3, figsize=(15, 4 * top_n))
    if top_n == 1:
        axes = axes.reshape(1, -1)

    for idx, model_name in enumerate(top_models):
        model = results[model_name]["model"]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # Residuals vs Fitted
        ax = axes[idx, 0]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{model_name}: Residuals vs Fitted")

        # Q-Q Plot
        ax = axes[idx, 1]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"{model_name}: Q-Q Plot")

        # Residual Histogram
        ax = axes[idx, 2]
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(residuals.mean(), color="red", linestyle="--", label=f"Mean: {residuals.mean():.0f}")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model_name}: Residual Distribution")
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Residual plots saved to: {save_path}")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ["MAE", "RMSE", "R2", "MAPE"],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot model comparison bar charts.

    Args:
        comparison_df: DataFrame from compare_models
        metrics: Metrics to plot
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        if metric in comparison_df.columns:
            # Sort by metric (ascending for error metrics, descending for R2)
            ascending = metric != "R2"
            plot_df = comparison_df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)

            bars = ax.barh(plot_df["model"], plot_df[metric])
            ax.set_xlabel(metric)
            ax.set_title(f"Model Comparison: {metric}")

            # Add value labels
            for bar, val in zip(bars, plot_df[metric]):
                ax.text(bar.get_width() + 0.01 * bar.get_width(),
                       bar.get_y() + bar.get_height()/2,
                       f"{val:,.2f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Model comparison plot saved to: {save_path}")

    return fig


def plot_feature_importance_comparison(
    results: Dict,
    feature_names: List[str],
    top_k: int = 15,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Compare feature importance across models.

    Args:
        results: Training results
        feature_names: List of feature names
        top_k: Number of top features to show
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Get importance for each model
    importance_data = []

    for name, result in results.items():
        if "error" in result:
            continue

        model = result["model"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            continue

        imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
        imp_df = imp_df.sort_values("importance", ascending=False).head(top_k)
        imp_df["model"] = name
        importance_data.append(imp_df)

    if not importance_data:
        print("No feature importance available for comparison")
        return None

    combined = pd.concat(importance_data, ignore_index=True)

    # Pivot for heatmap
    pivot = combined.pivot(index="feature", columns="model", values="importance")
    pivot = pivot.fillna(0)

    # Reorder features by mean importance
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop("mean", axis=1)

    fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.4)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Importance"})
    ax.set_title("Feature Importance Comparison Across Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Feature importance comparison saved to: {save_path}")

    return fig


def plot_learning_curves(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
    cv: int = 5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot learning curves for bias-variance diagnosis.

    Args:
        model: Model to evaluate
        X: Features
        y: Target
        train_sizes: Relative training set sizes
        cv: Cross-validation folds
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    from sklearn.model_selection import learning_curve

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv,
        scoring="neg_root_mean_squared_error", n_jobs=-1
    )

    train_rmse = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_rmse = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes_abs, train_rmse, "o-", label="Training RMSE", color="blue")
    ax.fill_between(train_sizes_abs, train_rmse - train_std, train_rmse + train_std, alpha=0.2, color="blue")
    ax.plot(train_sizes_abs, val_rmse, "o-", label="Validation RMSE", color="red")
    ax.fill_between(train_sizes_abs, val_rmse - val_std, val_rmse + val_std, alpha=0.2, color="red")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("RMSE")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning curves saved to: {save_path}")

    return fig


def generate_evaluation_report(
    results: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    output_dir: Path = REPORTS_DIR
) -> Dict:
    """
    Generate comprehensive evaluation report.

    Args:
        results: Training results
        X_test: Test features
        y_test: Test target
        feature_names: Feature names
        output_dir: Output directory

    Returns:
        Dictionary with all evaluation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Evaluation Report ===")

    # 1. Model comparison
    print("\n1. Comparing models on test set...")
    comparison_df = compare_models(results, X_test, y_test)

    # Save comparison
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Model comparison saved to: {comparison_path}")

    # 2. Detailed evaluation for each model
    print("\n2. Detailed evaluation...")
    detailed_results = {}
    for name, result in results.items():
        if "error" in result:
            continue
        model = result["model"]
        detailed_results[name] = evaluate_model_detailed(model, X_test, y_test, name)

    # 3. Generate plots
    print("\n3. Generating plots...")
    plot_residuals(results, X_test, y_test, save_path=output_dir / "residuals.png")
    plot_model_comparison(comparison_df, save_path=output_dir / "model_comparison.png")
    plot_feature_importance_comparison(results, feature_names, save_path=output_dir / "feature_importance.png")

    # Learning curves for best model
    best_name = comparison_df.iloc[0]["model"]
    best_model = results[best_name]["model"]
    plot_learning_curves(best_model, X_test, y_test, save_path=output_dir / f"learning_curves_{best_name}.png")

    # 4. Generate markdown report
    print("\n4. Generating markdown report...")
    report_path = output_dir / "model_comparison.md"
    with open(report_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Best Model:** {best_name}\n\n")

        f.write("## Test Set Performance\n\n")
        f.write(comparison_df[["model", "MAE", "RMSE", "R2", "MAPE"]].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Cross-Validation Results\n\n")
        cv_cols = ["model", "cv_rmse_mean", "cv_rmse_std"]
        f.write(comparison_df[cv_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Residual Diagnostics\n\n")
        for name in comparison_df.head(4)["model"]:
            if name in detailed_results:
                dr = detailed_results[name]
                f.write(f"### {name}\n")
                f.write(f"- Residual Mean: {dr['residual_stats']['mean']:.2f}\n")
                f.write(f"- Residual Std: {dr['residual_stats']['std']:.2f}\n")
                f.write(f"- Shapiro-Wilk p-value: {dr['normality_tests']['shapiro_p']:.4f}\n")
                f.write(f"- Breusch-Pagan p-value: {dr['heteroscedasticity'].get('p_value', 'N/A'):.4f}\n")
                f.write(f"- 95% Prediction Interval: ±${dr['prediction_intervals']['95%']:,.0f}\n\n")

    print(f"Markdown report saved to: {report_path}")

    # 5. Save detailed results
    joblib.dump(detailed_results, output_dir / "detailed_evaluation.pkl")

    print("\n=== Evaluation Report Complete ===")
    return {
        "comparison": comparison_df,
        "detailed": detailed_results,
        "best_model": best_name
    }


if __name__ == "__main__":
    from src.data.preprocessing import load_processed_data
    from src.models.train import load_training_results

    X_train, X_test, y_train, y_test, scaler = load_processed_data()
    results = load_training_results()

    feature_names = X_train.columns.tolist()
    eval_results = generate_evaluation_report(results, X_test, y_test, feature_names)