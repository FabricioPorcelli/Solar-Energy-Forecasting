# Solar Power Generation Forecasting with Machine Learning

## Overview
This project builds a production-style ML pipeline to forecast solar power generation (kW) using historical weather and temporal data. The goal is to support grid stability and renewable integration with accurate, explainable predictions.

## Problem Statement
Accurate solar energy forecasting is essential for grid operators and energy planners to manage supply, demand, and storage. This project addresses the regression problem of predicting hourly solar power output based on past weather and time features.

## Data
- ~3000 hourly observations
- **Target:** Power generated (kW), renamed internally as `power_generated_kw`
- **Features:**
  - Temporal: Year, Month, Day, Day of Year, First Hour of Period
  - Solar: Distance to Solar Noon, Is Daylight
  - Weather: Temperature, Wind Speed, Wind Direction, Sky Cover, Humidity, Pressure, Visibility

## Methodology
- Data cleaning and preprocessing ([src/preprocessing.py](src/preprocessing.py))
- Feature engineering (cyclic encoding, solar geometry) ([src/features.py](src/features.py))
- TimeSeriesSplit cross-validation (no random split)
- Model comparison: Baseline, Linear Regression, Random Forest, HistGradientBoostingRegressor
- Main metric: MAE (Mean Absolute Error)
- Final model: HistGradientBoostingRegressor (scikit-learn)
- Baseline: Previous value (naive last-value)

## Results
- **Best Model:** HistGradientBoostingRegressor
- **MAE:** ~1900 kW
- **RMSE:** ~X.XX kW
- Outperforms naive baseline and linear models

## Deployment
- Interactive [Streamlit app](app/streamlit_app.py) for real-time prediction and visualization
- Model and feature importances saved with joblib and CSV

## Tech Stack
- Python 3.11
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit
- joblib

## Project Structure
- `src/`: Pipeline scripts (preprocessing, features, training, evaluation)
- `data/`: Raw and processed data
- `models/`: Trained models and feature importances
- `app/`: Streamlit application

---

**For details, see the code and documentation in each module.**