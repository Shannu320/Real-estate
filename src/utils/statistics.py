"""Statistical analysis utilities."""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


def descriptive_statistics(series: pd.Series) -> Dict:
    """
    Comprehensive descriptive statistics.

    Args:
        series: Pandas Series

    Returns:
        Dictionary with statistics
    """
    s = series.dropna()

    return {
        "count": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else None,
        "std": s.std(),
        "var": s.var(),
        "min": s.min(),
        "max": s.max(),
        "range": s.max() - s.min(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "iqr": s.quantile(0.75) - s.quantile(0.25),
        "skew": s.skew(),
        "kurtosis": s.kurtosis(),
        "cv": s.std() / s.mean() if s.mean() != 0 else np.nan,
        "sem": s.sem(),
        "mad": (s - s.median()).abs().median(),
        "percentiles": {
            f"p{p}": s.quantile(p/100) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
    }


def normality_tests(series: pd.Series, sample_size: int = 5000) -> Dict:
    """
    Test for normality.

    Args:
        series: Data to test
        sample_size: Max sample size for Shapiro-Wilk

    Returns:
        Dictionary with test results
    """
    s = series.dropna()
    n = min(len(s), sample_size)
    sample = s.sample(n, random_state=42)

    results = {}

    # Shapiro-Wilk
    try:
        stat, p = stats.shapiro(sample)
        results["shapiro_wilk"] = {"statistic": stat, "p_value": p, "normal": p > 0.05}
    except Exception as e:
        results["shapiro_wilk"] = {"error": str(e)}

    # Jarque-Bera
    try:
        stat, p = stats.jarque_bera(s)
        results["jarque_bera"] = {"statistic": stat, "p_value": p, "normal": p > 0.05}
    except Exception as e:
        results["jarque_bera"] = {"error": str(e)}

    # D'Agostino's K^2
    try:
        stat, p = stats.normaltest(s)
        results["dagostino"] = {"statistic": stat, "p_value": p, "normal": p > 0.05}
    except Exception as e:
        results["dagostino"] = {"error": str(e)}

    # Anderson-Darling
    try:
        result = stats.anderson(s, dist="norm")
        results["anderson_darling"] = {
            "statistic": result.statistic,
            "critical_values": dict(zip(result.significance_level, result.critical_values)),
            "normal": result.statistic < result.critical_values[2]  # 5% level
        }
    except Exception as e:
        results["anderson_darling"] = {"error": str(e)}

    return results


def calculate_vif(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for features.

    Args:
        df: DataFrame with features
        features: List of feature names. If None, use all numeric.

    Returns:
        DataFrame with VIF values
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    vif_data = df[features].dropna()

    if len(vif_data) <= len(features):
        raise ValueError("Not enough samples for VIF calculation")

    vif_results = []
    for i, feature in enumerate(features):
        try:
            vif = variance_inflation_factor(vif_data.values, i)
            vif_results.append({"feature": feature, "VIF": vif})
        except:
            vif_results.append({"feature": feature, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_results).sort_values("VIF", ascending=False)
    vif_df["high_vif"] = vif_df["VIF"] > 5
    vif_df["very_high_vif"] = vif_df["VIF"] > 10

    return vif_df


def residual_analysis(residuals: np.ndarray, X: Optional[pd.DataFrame] = None) -> Dict:
    """
    Comprehensive residual analysis.

    Args:
        residuals: Model residuals
        X: Features (for heteroscedasticity tests)

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Basic statistics
    results["statistics"] = descriptive_statistics(pd.Series(residuals))

    # Normality
    results["normality"] = normality_tests(pd.Series(residuals))

    # Autocorrelation (Durbin-Watson)
    try:
        dw = durbin_watson(residuals)
        results["durbin_watson"] = {
            "statistic": dw,
            "interpretation": "positive autocorrelation" if dw < 1.5 else
                             "negative autocorrelation" if dw > 2.5 else
                             "no autocorrelation"
        }
    except Exception as e:
        results["durbin_watson"] = {"error": str(e)}

    # Heteroscedasticity tests
    if X is not None:
        try:
            bp_lm, bp_p, _, _ = het_breuschpagan(residuals, X)
            results["breusch_pagan"] = {
                "lm_statistic": bp_lm,
                "p_value": bp_p,
                "heteroscedastic": bp_p < 0.05
            }
        except Exception as e:
            results["breusch_pagan"] = {"error": str(e)}

        try:
            white_lm, white_p, _, _ = het_white(residuals, X)
            results["white_test"] = {
                "lm_statistic": white_lm,
                "p_value": white_p,
                "heteroscedastic": white_p < 0.05
            }
        except Exception as e:
            results["white_test"] = {"error": str(e)}

    return results


def outlier_analysis(series: pd.Series, method: str = "iqr", factor: float = 1.5) -> Dict:
    """
    Analyze outliers in a series.

    Args:
        series: Data to analyze
        method: 'iqr' or 'zscore'
        factor: Multiplier for IQR or z-score threshold

    Returns:
        Dictionary with outlier analysis
    """
    s = series.dropna()

    if method == "iqr":
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = s[(s < lower) | (s > upper)]
        bounds = {"lower": lower, "upper": upper, "method": "IQR"}
    else:
        mean = s.mean()
        std = s.std()
        lower = mean - factor * std
        upper = mean + factor * std
        outliers = s[(s < lower) | (s > upper)]
        bounds = {"lower": lower, "upper": upper, "method": "z-score"}

    return {
        "n_outliers": len(outliers),
        "outlier_percentage": len(outliers) / len(s) * 100,
        "bounds": bounds,
        "outlier_values": outliers.tolist(),
        "outlier_indices": outliers.index.tolist()
    }


def correlation_analysis(df: pd.DataFrame, target: str, method: str = "pearson") -> pd.DataFrame:
    """
    Analyze correlations with target.

    Args:
        df: DataFrame
        target: Target column
        method: Correlation method

    Returns:
        DataFrame with correlations sorted by absolute value
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target not in numeric_cols:
        raise ValueError(f"Target {target} not in numeric columns")

    correlations = df[numeric_cols].corr(method=method)[target].drop(target)
    corr_df = pd.DataFrame({
        "feature": correlations.index,
        "correlation": correlations.values,
        "abs_correlation": np.abs(correlations.values)
    }).sort_values("abs_correlation", ascending=False)

    # Add significance (approximate)
    n = len(df)
    corr_df["t_stat"] = corr_df["correlation"] * np.sqrt((n - 2) / (1 - corr_df["correlation"]**2))
    corr_df["p_value"] = 2 * (1 - stats.t.cdf(np.abs(corr_df["t_stat"]), n - 2))
    corr_df["significant_05"] = corr_df["p_value"] < 0.05
    corr_df["significant_01"] = corr_df["p_value"] < 0.01

    return corr_df


def group_statistics(df: pd.DataFrame, groupby: str, target: str) -> pd.DataFrame:
    """
    Calculate statistics for target grouped by categorical variable.

    Args:
        df: DataFrame
        groupby: Column to group by
        target: Target column

    Returns:
        DataFrame with group statistics
    """
    grouped = df.groupby(groupby)[target].agg([
        "count", "mean", "median", "std", "min", "max",
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ]).round(2)

    grouped.columns = ["count", "mean", "median", "std", "min", "max", "q25", "q75"]
    grouped["iqr"] = grouped["q75"] - grouped["q25"]
    grouped["cv"] = (grouped["std"] / grouped["mean"]).round(4)

    return grouped.sort_values("mean", ascending=False)


def anova_test(df: pd.DataFrame, groupby: str, target: str) -> Dict:
    """
    One-way ANOVA test.

    Args:
        df: DataFrame
        groupby: Grouping column
        target: Target column

    Returns:
        Dictionary with ANOVA results
    """
    groups = [group[target].values for name, group in df.groupby(groupby)]
    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        return {"error": "Need at least 2 groups with data"}

    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size (eta squared)
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((all_values - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "eta_squared": eta_squared,
        "effect_size": "large" if eta_squared > 0.14 else "medium" if eta_squared > 0.06 else "small"
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        data: Input data
        statistic: Function to compute statistic
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed

    Returns:
        (lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    n = len(data)
    boot_stats = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)

    return lower, upper


def compare_distributions(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = "mannwhitney"
) -> Dict:
    """
    Compare two distributions.

    Args:
        group1: First group
        group2: Second group
        test: Test to use ('mannwhitney', 'ttest', 'ks')

    Returns:
        Dictionary with test results
    """
    results = {}

    if test == "mannwhitney":
        stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        results["test"] = "Mann-Whitney U"
        results["statistic"] = stat
        results["p_value"] = p

    elif test == "ttest":
        stat, p = stats.ttest_ind(group1, group2, equal_var=False)
        results["test"] = "Welch's t-test"
        results["statistic"] = stat
        results["p_value"] = p

    elif test == "ks":
        stat, p = stats.ks_2samp(group1, group2)
        results["test"] = "Kolmogorov-Smirnov"
        results["statistic"] = stat
        results["p_value"] = p

    results["significant"] = results["p_value"] < 0.05

    # Effect size (Cohen's d for t-test, rank-biserial for Mann-Whitney)
    if test == "ttest":
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        results["effect_size"] = d
        results["effect_size_interpretation"] = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"

    return results


if __name__ == "__main__":
    from src.data.ingestion import load_raw_data

    df = load_raw_data()

    # Test statistics
    print("=== Descriptive Statistics for Price ===")
    desc = descriptive_statistics(df["price"])
    for k, v in desc.items():
        if k != "percentiles":
            print(f"  {k}: {v}")

    print("\n=== Normality Tests for Price ===")
    norm = normality_tests(df["price"])
    for test, res in norm.items():
        if "error" not in res:
            print(f"  {test}: p={res['p_value']:.4f}, normal={res['normal']}")

    print("\n=== Correlation with Price ===")
    corr = correlation_analysis(df, "price")
    print(corr.head(10).to_string(index=False))

    print("\n=== ANOVA: Price by Waterfront ===")
    anova = anova_test(df, "waterfront", "price")
    print(f"  F={anova['f_statistic']:.2f}, p={anova['p_value']:.4f}, eta²={anova['eta_squared']:.4f}")