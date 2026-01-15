"""
Feature engineering for solar power forecasting.
- Cyclic encoding for hour and day of year
- Keep is_daylight as binary
- Drop redundant temporal columns after encoding
"""
import pandas as pd
import numpy as np

PROCESSED_PATH = "data/processed/solar_clean.csv"
FEATURES_PATH = "data/processed/solar_features.csv"
TARGET_COL = "power_generated_kw"

def load_data(path):
    return pd.read_csv(path)

def encode_cyclic(df, col, max_val):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df

def feature_engineering(df):
    df = encode_cyclic(df, "first_hour_of_period", 24)
    df = encode_cyclic(df, "day_of_year", 366)

    df["distance_to_noon_squared"] = df["distance_to_solar_noon"] ** 2

    df["solar_potential"] = (
        (1 - df["sky_cover"] / 100)
        * (1 - df["distance_to_solar_noon"] / df["distance_to_solar_noon"].max())
    )

    if "is_daylight" in df.columns:
        df["is_daylight"] = df["is_daylight"].fillna(0).astype(int)

    drop_cols = ["year", "month", "day", "first_hour_of_period"] #, "day_of_year"
    drop_cols = [c for c in drop_cols if c in df.columns and c != TARGET_COL]
    df = df.drop(columns=drop_cols)

    return df

def main():
    df = load_data(PROCESSED_PATH)
    df = feature_engineering(df)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Saved features to {FEATURES_PATH}")

if __name__ == "__main__":
    main()
