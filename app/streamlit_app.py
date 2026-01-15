"""
Streamlit app for solar power generation forecasting
Phase 6 — Interactive demo for non-technical users
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../models/final_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "../data/processed/solar_features.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "../models/feature_importance.csv")

# ---------------- MATPLOTLIB STYLE ----------------
width, height = 8, 4
scale = width / 7
title_fs = 10 * scale
label_fs = 9 * scale
axis_fs = 9 * scale

# Colors
dots_color = "#F4B400"
curve_color = "#E65100"
shadow_color = "#FF8686"
# ---------------- STREAMLIT STYLE ----------------
st.set_page_config(
    page_title="Solar Power Forecasting",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Main app container */
        .block-container {
            max-width: 1000px;
            padding-left: 1rem;
            padding-right: 1rem;
            margin: auto;
        }

        /* Center charts */
        canvas {
            max-width: 100% !important;
        }

        /* Reduce header spacing */
        h1, h2, h3 {
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# ---------------- LOADERS ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH)


model = load_model()
df_features = load_features()

# Get model name for display
if hasattr(model, 'named_steps') and 'model' in model.named_steps:
    model_name = type(model.named_steps['model']).__name__
else:
    model_name = type(model).__name__

# ---------------- HEADER ----------------
st.title("☀️ Solar Power Generation Forecasting")

st.markdown("""
This application predicts **hourly solar power generation** using weather and time-based features.

**Technical summary**
- Model: `{}`
- Validation: `TimeSeriesSplit`
- Metric: `Mean Absolute Error (MAE)`
""".format(model_name))

# ---------------- USER INPUTS ----------------
st.header("🔧 Input Conditions")

col1, col2 = st.columns(2)

with col1:
    # Cargar importancias como diccionario
    importance_dict = pd.read_csv(IMPORTANCE_PATH, index_col=0, header=None)[1].to_dict()

    def label_with_importance(label, feature, importance_dict):
        imp = importance_dict.get(feature, None)
        if imp is not None:
            return f"{label} (Imp: {imp:.2f})"
        else:
            return label

    hour = st.slider(
        label=label_with_importance("**Hour of Day**", "first_hour_of_period_cos", importance_dict),
        min_value=0, max_value=23, value=12, key="hour",
        format=None,
        help=None,
        disabled=False,
        label_visibility="visible"
    )

    day_of_year = st.slider(
        label=label_with_importance("**Day of Year**", "day_of_year_cos", importance_dict),
        min_value=1, max_value=366, value=180, key="doy",
        format=None,
        help=None,
        disabled=False,
        label_visibility="visible"
    )

    is_daylight = st.selectbox(
        label=label_with_importance("**Is Daylight?**", "is_daylight", importance_dict),
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No",
        key="is_daylight"
    )

    distance_to_noon = st.slider(
        label=label_with_importance("**Distance to Solar Noon (hours)**", "distance_to_solar_noon", importance_dict),
        min_value=0.0, max_value=12.0, value=2.0, key="dist_noon",
        format=None,
        help=None,
        disabled=False,
        label_visibility="visible"
    )

    sky_cover = st.slider(
        label=label_with_importance("**Sky Cover (0 = clear, 4 = overcast)**", "sky_cover", importance_dict),
        min_value=0, max_value=4, value=2, key="sky_cover"
    )

    humidity = st.slider(
        label=label_with_importance("**Relative Humidity (%)**", "relative_humidity", importance_dict),
        min_value=0, max_value=100, value=70, key="humidity"
    )

with col2:
    temperature = st.number_input(
        label=label_with_importance("**Average Temperature (°F)**", "average_temperature_day", importance_dict),
        value=60.0, key="temp"
    )

    wind_speed = st.number_input(
        label=label_with_importance("**Average Wind Speed (mph)**", "average_wind_speed_day", importance_dict),
        value=10.0, key="wind_speed"
    )

    wind_direction = st.number_input(
        label=label_with_importance("**Average Wind Direction (1–36)**", "average_wind_direction_day", importance_dict),
        min_value=1, max_value=36, value=18, key="wind_dir"
    )

    pressure = st.number_input(
        label=label_with_importance("**Barometric Pressure (inHg)**", "average_barometric_pressure_period", importance_dict),
        value=30.0, key="pressure"
    )

    visibility = st.number_input(
        label=label_with_importance("**Visibility (miles)**", "visibility", importance_dict),
        value=10.0, key="visibility"
    )

# ---------------- FEATURE ENGINEERING ----------------
def encode_cyclic(val, max_val):
    return np.sin(2 * np.pi * val / max_val), np.cos(2 * np.pi * val / max_val)

template_row = df_features.iloc[0].copy()

template_row.update({
    "is_daylight": is_daylight,
    "distance_to_solar_noon": distance_to_noon,
    "average_temperature_day": temperature,
    "average_wind_direction_day": wind_direction,
    "average_wind_speed_day": wind_speed,
    "average_wind_speed_period": wind_speed,
    "sky_cover": sky_cover,
    "visibility": visibility,
    "relative_humidity": humidity,
    "average_barometric_pressure_period": pressure
})

hour_sin, hour_cos = encode_cyclic(hour, 24)
doy_sin, doy_cos = encode_cyclic(day_of_year, 366)

template_row["first_hour_of_period_sin"] = hour_sin
template_row["first_hour_of_period_cos"] = hour_cos
template_row["day_of_year_sin"] = doy_sin
template_row["day_of_year_cos"] = doy_cos

input_df = pd.DataFrame([template_row.drop("power_generated_kw", errors="ignore")])

# ---------------- PREDICTION ----------------
st.header("📈 Prediction")

if st.button("Predict Power Generation"):
    pred = max(0, model.predict(input_df)[0])
    st.success(f"**Predicted Power Generated:** {pred:,.0f} kW")


# ------------------ GRAPHS ------------------
st.header("📊 Model Behavior")

# -------- Power Range --------

# Actual dataset hours (every 3 hours)
plot_hours = np.array([1, 4, 7, 10, 13, 16, 19, 22])

percentiles = [0.25, 0.5, 0.75]
curves = {p: [] for p in percentiles}

for h in plot_hours:
    row = template_row.copy()
    hs, hc = encode_cyclic(h, 24)
    row["first_hour_of_period_sin"] = hs
    row["first_hour_of_period_cos"] = hc

    for p in percentiles:
        row["sky_cover"] = df_features["sky_cover"].quantile(p)
        curves[p].append(
            max(0, model.predict(pd.DataFrame([row.drop("power_generated_kw", errors="ignore")]))[0])
        )

fig, ax = plt.subplots(figsize=(width, height))
ax.plot(plot_hours, curves[0.5], label="Median scenario", color=curve_color)
ax.fill_between(plot_hours, curves[0.25], curves[0.75], alpha=0.3, label="IQR", color=shadow_color)
ax.set_xlabel("Hour of Day", fontsize=axis_fs)
ax.set_ylabel("Predicted Power (kW)", fontsize=axis_fs)
ax.set_title("Expected Power Range vs Hour", fontsize=title_fs)
ax.legend()
ax.grid(True, alpha=0.2)
st.pyplot(fig)

# -------- Yearly Evolution of Predicted Power (Smoothed) --------

# Predict for each row, aggregate by day
df_pred = df_features.copy()
if 'power_generated_kw' in df_pred.columns:
    X_pred = df_pred.drop(columns=['power_generated_kw'])
else:
    X_pred = df_pred
df_pred['predicted_power'] = np.maximum(0, model.predict(X_pred))

# Aggregate by day of year
if 'day_of_year' in df_pred.columns:
    daily_pred = df_pred.groupby('day_of_year')['predicted_power'].mean().reset_index()
    x = daily_pred['day_of_year']
    y = daily_pred['predicted_power']
    xlabel = 'Day of Year'
else:
    x = df_pred.iloc[:, 0]
    y = df_pred['predicted_power']
    xlabel = df_pred.columns[0]

# Apply LOWESS smoothing
smoothed = lowess(y, x, frac=0.05, return_sorted=True)

fig, ax = plt.subplots(figsize=(width, height))
ax.scatter(x, y, color=dots_color, alpha=0.3, s=10, label="Daily Mean (raw)")
ax.plot(smoothed[:, 0], smoothed[:, 1], color=curve_color, linewidth=2.2, label="Year Trend (LOWESS)")
ax.set_xlabel(xlabel, fontsize=axis_fs)
ax.set_ylabel("Predicted Power (kW)", fontsize=axis_fs)
ax.set_title("Smoothed Solar Power Generation Trend Over the Year", fontsize=title_fs)
ax.grid(True, alpha=0.2)
ax.legend()
st.pyplot(fig)

# ---------------- FEATURE IMPORTANCE ----------------

st.header("🔍 What drives the prediction?")

importance = pd.read_csv(IMPORTANCE_PATH, index_col=0, header=None)
importance.columns = ["importance"]
importance = importance.dropna(axis=0, subset=["importance"])
importance = importance[importance.index.notnull()]
top10 = importance.head(10).sort_values("importance")

fig, ax = plt.subplots(figsize=(4, 3))
top10.plot(
    kind="barh", ax=ax, legend=False, color=curve_color
)
ax.set_title("Top 10 Feature Importances", fontsize=title_fs)
ax.set_xlabel("Relative importance", fontsize=axis_fs)
st.pyplot(fig)

st.header("💡 Some Insights")
st.markdown("""
Multiple algorithms are evaluated (including tree ensembles and linear models), and the best one is automatically selected based on validation MAE.

**Top features driving the prediction:**
- **`first_hour_of_period_cos`** and **`solar_potential`** (solar geometry)
- **`distance_to_solar_noon`** and **`distance_to_noon_squared`** (solar geometry)
- **`sky_cover`** (main weather-related driver)

Other relevant features include:
- **is_daylight**, **day_of_year_cos**, **relative_humidity**, **first_hour_of_period_sin**, **average_wind_direction_day**

Features like **average_temperature_day**, **visibility**, and **average_barometric_pressure_period** have very low importance in this model.

**Key insight:**
Solar geometry (hour, solar potential, distance to solar noon, day of year) overwhelmingly dominates the prediction, reflecting the physical reality of solar energy. Weather variables such as sky cover and humidity provide secondary, but meaningful, adjustments to the forecast.
""")

st.header("⚙️ Technical Model Behavior")
st.markdown(f"""
**Model type:** {model_name} (best of several tested)

- The model is selected from a pool of algorithms (tree ensembles, linear, boosting, etc.) using time series cross-validation and MAE.
- It learns patterns from historical data, using both temporal (e.g., day of year, hour) and weather features.
- It is robust to outliers and can capture non-linear relationships, but does not extrapolate far beyond the range of the training data.
- Most predictions fall within the typical range of observed power values (up to ~12,000 kW), even though the dataset contains rare, much higher values. This is expected for tree-based models, which average over many trees and are conservative with rare/extreme cases.
- The model is most sensitive to features that describe the position of the sun (solar_potential, hour, day of year), which aligns with the physics of solar power generation.
- Weather features (sky cover, humidity, wind) act as modifiers, reducing or increasing the predicted power depending on atmospheric conditions.
- The model does not predict negative power, and outputs are clipped at zero.

**Practical implication:**
The model provides reliable forecasts for typical days and conditions, but may underpredict rare, extreme peaks unless those are well-represented in the training data. For operational use, always compare predictions to historical extremes and consider uncertainty in rare scenarios.
""")
    
