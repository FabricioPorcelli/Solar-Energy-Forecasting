"""
Preprocessing script for solar power generation dataset.
Cleans, renames, sorts, and prepares data for modeling.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/raw/BigML_Dataset_5f50a4cc0d052e40e6000034.csv"
OUT_PATH = "data/processed/solar_clean.csv"

def load_data(path):
    """Load raw CSV data."""
    return pd.read_csv(path)

def rename_columns(df):
    """Rename columns to snake_case and remove parentheses."""
    df.columns = [
        col.lower().replace(" ", "_").replace("(", "").replace(")", "")
        for col in df.columns
    ]
    return df

def sort_chronologically(df):
    """Sort by year, day_of_year, first_hour_of_period."""
    return df.sort_values(["year", "day_of_year", "first_hour_of_period"]).reset_index(drop=True)

def convert_boolean_to_int(df):
    """Convert boolean columns to int (0/1)."""
    if "is_daylight" in df.columns:
        df["is_daylight"] = df["is_daylight"].astype(int)
    return df

def define_target(df):
    """Rename target column to power_generated_kw."""
    df.rename(columns={"power_generated": "power_generated_kw"}, inplace=True)
    return df

def handle_missing_wind_speed(df):
    """Drop or impute missing average_wind_speed_period (single row)."""
    if df["average_wind_speed_period"].isnull().sum() >= 1:
        # Impute with median
        median = df["average_wind_speed_period"].median()
        df["average_wind_speed_period"].fillna(median, inplace=True)
    return df

def main():
    df = load_data(RAW_PATH)
    df = rename_columns(df)
    df = sort_chronologically(df)
    df = convert_boolean_to_int(df)
    df = define_target(df)
    df = handle_missing_wind_speed(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data to {OUT_PATH}")

if __name__ == "__main__":
    main()
