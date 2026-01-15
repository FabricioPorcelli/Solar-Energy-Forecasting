import pandas as pd
from pathlib import Path

TARGET_COL = "power_generated_kw"

def load_data(path: Path):
    """
    Load features and target from processed CSV.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
