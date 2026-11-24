# Real Estate Price Intelligence & Market Valuation

A complete regression workflow from raw property data to a validated price-prediction model and interactive Power BI dashboard.

## Overview

This project implements a comprehensive property valuation system using the **King County House Sales dataset** (21,613 Seattle-area homes sold 2014-2015). The system estimates market prices and reveals how location, size, amenities, and property condition influence property value.

## Features

- **Data Ingestion**: Automated download and validation of Kaggle dataset
- **Exploratory Data Analysis**: Comprehensive EDA with visualizations and statistical analysis
- **Preprocessing**: Missing value handling, outlier detection (IQR), feature scaling, stratified train/test split
- **Feature Engineering**: Price per sqft, property age, amenity counts, location features, zipcode encoding
- **Regression Modeling**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM
- **Model Evaluation**: Cross-validation, hyperparameter tuning, residual analysis, prediction intervals
- **Power BI Dashboard**: Ready-to-use dataset and DAX measures for interactive market analysis

## Tech Stack

- **Python 3.10+** with virtual environment
- **Data**: pandas, numpy, scipy
- **ML**: scikit-learn, xgboost, lightgbm, statsmodels
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: Jupyter
- **Dashboard**: Power BI Desktop

## Project Structure

```
real estate/
├── data/
│   ├── raw/                 # Original downloaded datasets
│   ├── processed/           # Cleaned, engineered datasets
│   └── external/            # Reference data
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_powerbi_prep.ipynb
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       ├── visualization.py
│       └── statistics.py
├── powerbi/
│   ├── dax_measures.txt
│   ├── model_performance.csv
│   ├── feature_importance.csv
│   └── dashboard_structure.md
├── models/
│   ├── ridge.pkl
│   ├── lasso.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl          # Best model (R²=0.943, RMSE=$47.5K)
│   ├── best_model.pkl       # Alias to xgboost.pkl
│   ├── scaler.pkl           # StandardScaler fitted on raw features
│   └── feature_names.pkl    # Model feature names
├── reports/
│   ├── eda_report.md
│   ├── model_comparison.md
│   ├── model_comparison.csv
│   ├── valuation_output.csv
│   └── figures/
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate
cd real-estate

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

Execute notebooks in order:

```bash
jupyter notebook
```

Or run individual modules:

```bash
# Data ingestion
python -m src.data.ingestion

# Preprocessing
python -m src.data.preprocessing

# Feature engineering
python -m src.data.feature_engineering

# Model training
python -m src.models.train

# Model evaluation
python -m src.models.evaluate

# Prediction
python -m src.models.predict --interactive
```

### 3. Power BI Dashboard

1. Open Power BI Desktop
2. Load `data/processed/kc_house_for_powerbi.csv`
3. Copy DAX measures from `powerbi/dax_measures.txt`
4. Build dashboard following `powerbi/dashboard_structure.md`

## Key Results

### Model Performance (Test Set)

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|-----|------|
| **XGBoost** | **$30,987** | **$47,542** | **0.943** | **7.6%** |
| Random Forest | $39,869 | $59,817 | 0.909 | 9.6% |
| Ridge Regression | $82,879 | $108,240 | 0.704 | 20.1% |

*Lightweight training settings used to prevent system overheating (3 models, 15 iterations, 3-fold CV)*

### Key Features (Feature Importance - XGBoost)

1. **latitude** (0.23) - Geographic location
2. **grade** (0.18) - Construction quality grade (1-13)
3. **sqft_living** (0.16) - Living area square footage
4. **amenity_count** (0.05) - Count of premium features
5. **dist_to_seattle_center** (0.04) - Distance to downtown
6. **zipcode_freq** (0.03) - Location frequency encoding
7. **view** (0.03) - View quality (0-4)
8. **longitude** (0.03) - Geographic location
9. **sqft_above** (0.03) - Above-ground square footage
10. **waterfront** (0.03) - Waterfront property (binary)

### Price Differentiation (Fixed)

The model now correctly differentiates across property types:

| Property Type | Predicted Price | 95% Prediction Interval |
|---------------|-----------------|------------------------|
| Ultra Luxury Waterfront | **$3.05M** | $2.89M - $3.22M |
| High View No Waterfront | **$1.18M** | $1.02M - $1.35M |
| Average Seattle Home | **$940K** | $774K - $1.11M |
| New Construction Suburban | **$947K** | $780K - $1.11M |
| Renovated Mid-Century | **$1.05M** | $879K - $1.21M |
| Budget Older Home | **$258K** | $92K - $425K |

## Dataset

**King County House Sales** (Kaggle)
- 21,613 records
- 21 original features
- Sale dates: May 2014 - May 2015
- Target: Sale price (USD)

### Original Features
- `id`, `date`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`
- `waterfront`, `view`, `condition`, `grade`, `sqft_above`, `sqft_basement`
- `yr_built`, `yr_renovated`, `zipcode`, `lat`, `long`
- `sqft_living15`, `sqft_lot15`

### Engineered Features (29 total)
- `price_per_sqft` - Price per square foot
- `age` - Property age at sale
- `is_renovated` - Renovation flag
- `age_band` - Age categories (0-10, 10-20, 20-40, 40-60, 60+)
- `amenity_count` - Count of premium features (waterfront, view, condition≥4, grade≥8)
- `has_basement` - Basement presence
- `living_ratio` - Living area / lot area
- `zipcode_freq` - Zipcode frequency encoding
- `dist_to_seattle_center` - Distance to downtown Seattle (miles)
- `sale_year`, `sale_month`, `sale_quarter`, `sale_dayofweek` - Temporal features

## Deliverables

1. **Prepared Dataset** - Raw, processed, and engineered versions
   - `data/processed/kc_house_engineered_raw.csv` - 29 raw engineered features
   - `data/processed/kc_house_for_powerbi.csv` - 40 columns with predictions
2. **EDA Report** - `reports/eda_report.md` with visualizations
3. **Regression Models** - 7 trained models with hyperparameter tuning
   - `models/xgboost.pkl` (best), `models/random_forest.pkl`, `models/ridge.pkl`, etc.
4. **Model Comparison** - `powerbi/model_performance.csv` with statistical validation
5. **Property Valuation Output** - `data/processed/kc_house_for_powerbi.csv` with 21K predictions
6. **Power BI Dashboard** - Ready-to-import dataset and DAX measures
   - `powerbi/dax_measures.txt` - 40+ DAX measures
   - `powerbi/dashboard_structure.md` - 4-page layout guide
   - `powerbi/feature_importance.csv` - Feature importance across models

## Usage Examples

### Interactive Prediction

```bash
python -m src.models.predict --interactive
```

### Batch Prediction from CSV

```bash
python -m src.models.predict --input properties.csv --output predictions.csv
```

### Programmatic Usage

```python
from src.models.predict import load_model, load_scaler, predict_price

model = load_model('best_model')
scaler = load_scaler()

property_data = {
    'bedrooms': 3,
    'bathrooms': 2.5,
    'sqft_living': 2200,
    'sqft_lot': 8000,
    'floors': 2,
    'waterfront': 0,
    'view': 0,
    'condition': 4,
    'grade': 8,
    'yr_built': 1995,
    'yr_renovated': 2010,
    'zipcode': 98103,
    'lat': 47.65,
    'long': -122.35,
    'sqft_living15': 2100,
    'sqft_lot15': 7500
}

result = predict_price(model, scaler, property_data, feature_names)
print(f"Predicted: {result['predicted_price_formatted']}")
print(f"95% CI: {result['prediction_interval']['lower_formatted']} - {result['prediction_interval']['upper_formatted']}")
```

## Technical Notes

### Fixed: Price Differentiation Issue

**Problem**: Initial model predictions clustered around $800-900K regardless of property type.

**Root Cause**: Model trained on pre-scaled features (grade range: -4.7 to 3.8) but scaler fit on these pre-scaled values. Test cases with raw values (grade=13) became 13.04 - outside training range.

**Solution**: 
1. Created `kc_house_engineered_raw.csv` with 29 un-scaled features
2. Retrained XGBoost on raw features with StandardScaler fit on raw data
3. Updated `predict.py` to use raw engineered data for feature engineering

### Lightweight Training Configuration

To prevent system overheating during training:
- 3 models (Ridge, Random Forest, XGBoost) instead of 7
- 15 RandomizedSearchCV iterations (vs 50)
- 3-fold CV (vs 5-fold)
- 5,000 sample subset for hyperparameter tuning
- Full training data for final model fit

## Statistical Validation

- **Cross-Validation**: 3-fold CV on training set (lightweight)
- **Hyperparameter Tuning**: RandomizedSearchCV (15 iterations)
- **Residual Analysis**: Normality (Shapiro-Wilk), Heteroscedasticity (Breusch-Pagan), Autocorrelation (Durbin-Watson)
- **Multicollinearity**: VIF analysis (threshold > 5)
- **Prediction Intervals**: Bootstrap-based 95% and 80% intervals

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Repository

**GitHub**: [https://github.com/Shannu320/Real-estate](https://github.com/Shannu320/Real-estate)

## Contact

Prem Kumar Reddy - [GitHub](https://github.com/premkumarreddy)