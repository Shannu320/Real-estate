"""Visualization utilities for EDA and model analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def setup_plotting_style():
    """Configure global plotting style."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3
    })


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 50,
    log_scale: bool = False,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot distribution of a single variable."""
    fig, axes = plt.subplots(1, 2 if not log_scale else 3, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    ax.hist(df[column].dropna(), bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {column}")

    # Box plot
    ax = axes[1]
    ax.boxplot(df[column].dropna(), vert=False)
    ax.set_xlabel(column)
    ax.set_title(f"Box Plot of {column}")

    if log_scale:
        # Log scale histogram
        ax = axes[2]
        log_data = np.log1p(df[column].dropna())
        ax.hist(log_data, bins=bins, edgecolor="black", alpha=0.7)
        ax.set_xlabel(f"log({column})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Log Distribution of {column}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    annot: bool = True,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot correlation heatmap."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr = df[columns].corr(method=method)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=annot, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title(f"Correlation Heatmap ({method})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_target_vs_feature(
    df: pd.DataFrame,
    target: str,
    feature: str,
    plot_type: str = "scatter",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot target vs feature relationship."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "scatter":
        ax.scatter(df[feature], df[target], alpha=0.5, s=10)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)

    elif plot_type == "box":
        # For categorical features
        sns.boxplot(data=df, x=feature, y=target, ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)

    elif plot_type == "violin":
        sns.violinplot(data=df, x=feature, y=target, ax=ax)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)

    ax.set_title(f"{target} vs {feature}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_price_geographic(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "long",
    price_col: str = "price",
    save_path: Optional[Path] = None
) -> go.Figure:
    """Create interactive geographic price plot using Plotly."""
    # Use scattergeo instead of scatter_mapbox for broader compatibility
    fig = px.scatter_geo(
        df,
        lat=lat_col,
        lon=lon_col,
        color=price_col,
        size=price_col,
        color_continuous_scale="Viridis",
        size_max=15,
        projection="natural earth",
        hover_data=["price", "sqft_living", "bedrooms", "bathrooms", "zipcode"]
    )
    fig.update_layout(
        title="Property Prices - Geographic Distribution",
        height=700,
        geo=dict(
            scope="usa",
            center=dict(lat=47.6, lon=-122.3),
            projection_scale=50000
        )
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_price_trends(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    groupby_col: Optional[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot price trends over time."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_month"] = df[date_col].dt.to_period("M")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average price over time
    ax = axes[0, 0]
    monthly_avg = df.groupby("year_month")[price_col].mean()
    monthly_avg.plot(ax=ax, marker="o")
    ax.set_title("Average Price Over Time")
    ax.set_ylabel("Average Price ($)")
    ax.tick_params(axis="x", rotation=45)

    # Median price over time
    ax = axes[0, 1]
    monthly_median = df.groupby("year_month")[price_col].median()
    monthly_median.plot(ax=ax, marker="o", color="orange")
    ax.set_title("Median Price Over Time")
    ax.set_ylabel("Median Price ($)")
    ax.tick_params(axis="x", rotation=45)

    # Transaction volume
    ax = axes[1, 0]
    monthly_count = df.groupby("year_month").size()
    monthly_count.plot(ax=ax, kind="bar", color="steelblue")
    ax.set_title("Transaction Volume Over Time")
    ax.set_ylabel("Number of Sales")
    ax.tick_params(axis="x", rotation=45)

    # Price per sqft over time
    ax = axes[1, 1]
    if "price_per_sqft" in df.columns:
        monthly_ppsqft = df.groupby("year_month")["price_per_sqft"].mean()
        monthly_ppsqft.plot(ax=ax, marker="o", color="green")
        ax.set_title("Avg Price per SqFt Over Time")
        ax.set_ylabel("Price per SqFt ($)")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_categorical_analysis(
    df: pd.DataFrame,
    target: str,
    categorical_cols: List[str],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot target distribution across categorical variables."""
    n_cols = min(3, len(categorical_cols))
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        # Box plot for target by category
        sns.boxplot(data=df, x=col, y=target, ax=ax)
        ax.set_title(f"{target} by {col}")
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for idx in range(len(categorical_cols), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_pairwise_relationships(
    df: pd.DataFrame,
    columns: List[str],
    target: Optional[str] = None,
    save_path: Optional[Path] = None
) -> sns.PairGrid:
    """Plot pairwise relationships."""
    plot_cols = columns + ([target] if target and target not in columns else [])

    g = sns.pairplot(
        df[plot_cols].sample(min(1000, len(df)), random_state=42),
        hue=target if target in plot_cols else None,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20}
    )
    g.fig.suptitle("Pairwise Relationships", y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return g


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 20,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot feature importance horizontal bar chart."""
    plot_df = importance_df.head(top_k).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    bars = ax.barh(plot_df["feature"], plot_df["importance"])

    # Color by importance
    colors = plt.cm.YlOrRd(plot_df["importance"] / plot_df["importance"].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importance")

    # Add value labels
    for bar, val in zip(bars, plot_df["importance"]):
        ax.text(bar.get_width() + 0.01 * bar.get_width(),
               bar.get_y() + bar.get_height()/2,
               f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_eda_dashboard(
    df: pd.DataFrame,
    target: str = "price",
    save_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """Create comprehensive EDA dashboard with multiple plots."""
    figures = {}

    # 1. Target distribution
    print("Plotting target distribution...")
    fig = plot_distribution(df, target, log_scale=True)
    figures["target_distribution"] = fig
    if save_dir:
        fig.savefig(save_dir / "target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Correlation heatmap
    print("Plotting correlation heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig = plot_correlation_heatmap(df, numeric_cols)
    figures["correlation_heatmap"] = fig
    if save_dir:
        fig.savefig(save_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Key feature relationships
    print("Plotting key feature relationships...")
    key_features = ["sqft_living", "grade", "bedrooms", "bathrooms", "age", "waterfront"]
    key_features = [f for f in key_features if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features[:6]):
        ax = axes[idx]
        if df[feat].nunique() > 10:
            ax.scatter(df[feat], df[target], alpha=0.3, s=10)
        else:
            sns.boxplot(data=df, x=feat, y=target, ax=ax)
        ax.set_title(f"{target} vs {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel(target)

    plt.tight_layout()
    figures["key_relationships"] = fig
    if save_dir:
        fig.savefig(save_dir / "key_relationships.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Categorical analysis
    print("Plotting categorical analysis...")
    cat_cols = ["zipcode", "waterfront", "view", "condition", "grade", "bedrooms"]
    cat_cols = [c for c in cat_cols if c in df.columns and df[c].nunique() < 30]
    fig = plot_categorical_analysis(df, target, cat_cols)
    figures["categorical_analysis"] = fig
    if save_dir:
        fig.savefig(save_dir / "categorical_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Geographic plot (interactive)
    print("Creating geographic plot...")
    if "lat" in df.columns and "long" in df.columns:
        fig = plot_price_geographic(df)
        figures["geographic"] = fig
        if save_dir:
            fig.write_html(save_dir / "geographic_prices.html")

    print("EDA dashboard complete!")
    return figures


if __name__ == "__main__":
    from src.data.ingestion import load_raw_data

    df = load_raw_data()
    setup_plotting_style()

    # Quick EDA
    figs = create_eda_dashboard(df, target="price", save_dir=Path("reports/figures"))