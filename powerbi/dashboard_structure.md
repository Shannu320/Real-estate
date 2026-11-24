# POWER BI DASHBOARD - PROPERTY VALUATION

## Page 1: Market Overview
**KPI Cards:**
- Average Price
- Median Price 
- Avg Price per SqFt
- Total Properties
- % Waterfront
- % Renovated

**Visuals:**
- Map: Property locations colored by price (use lat/long)
- Line Chart: Average Price by Month (sale_year, sale_month)
- Bar Chart: Avg Price by Zipcode (top 20)
- Box Plot: Price Distribution by Property Type

## Page 2: Property Valuation Tool
**Input Parameters (What-if Parameters):**
- Bedrooms (1-8)
- Bathrooms (1-5)
- Sqft Living (500-10000)
- Grade (1-13)
- Condition (1-5)
- Waterfront (0/1)
- View (0-4)
- Year Built (1900-2015)
- Renovated (0/1)
- Zipcode (dropdown)

**Output:**
- Predicted Price (card)
- 95% Prediction Interval (card range)
- Similar Properties Table (filter by zipcode, bedrooms, sqft)
- Price vs Predicted Scatter (with identity line)

## Page 3: Model Performance
**KPI Cards:**
- Best Model RMSE
- Best Model R²
- Best Model MAPE
- Accuracy within 10%

**Visuals:**
- Bar Chart: Model Comparison (RMSE, MAE, R², MAPE)
- Residual Plots (Residuals vs Fitted, Q-Q Plot, Histogram)
- Feature Importance Heatmap (models x features)
- Learning Curves (Training vs Validation RMSE)
- Prediction Interval Coverage

## Page 4: Deep Dive Analysis
**Drill-through by Zipcode:**
- Zipcode-level price statistics
- Price trends over time
- Property characteristic profiles
- Model accuracy by zipcode

**Drill-through by Age Band:**
- Price per sqft by age band
- Renovation impact by age
- Condition distribution

**Drill-through by Condition/Grade:**
- Price distribution
- Feature correlations
- Prediction accuracy

## Slicers (Global):
- Date Range (sale_year, sale_month)
- Zipcode
- Bedrooms
- Property Type (Waterfront, View, Condition, Grade)
- Price Range
- Age Band

## Recommended Visual Types:
- Map: Azure Maps / Mapbox for geographic
- Line/Bar: Core visuals
- Scatter: Price vs Predicted, Price vs SqFt
- Box & Whisker: Price distributions
- Gauge: Accuracy %, R²
- Card: KPIs
- Table/Matrix: Detailed drill-through
