"""
train.py — Model training for solar power forecasting
"""

from pathlib import Path
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor

from data_loader import load_data
from baselines import naive_last_value_baseline
from model_selection import evaluate_model_cv

# External models
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
# Paths
DATA_PATH = Path("data/processed/solar_features.csv")
MODEL_PATH = Path("models/final_model.joblib")

# Config
N_SPLITS = 5
RANDOM_STATE = 42

def main():
    X, y = load_data(DATA_PATH)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # ---- Baseline ----
    baseline_mae = naive_last_value_baseline(y)
    print(f"Naive baseline (last value) MAE: {baseline_mae:.3f}")

    # ---- Models ----
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "use_scaler": True
        },
        "Ridge": {
            "model": Ridge(alpha=10.0, random_state=RANDOM_STATE),
            "use_scaler": True
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=RANDOM_STATE),
            "use_scaler": True
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_STATE
            ),
            "use_scaler": False
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_STATE
            ),
            "use_scaler": False
        },
        "HistGBR": {
            "model": HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            ),
            "use_scaler": False
        },
    }

    # Add external models if available
    if XGBRegressor is not None:
        models["XGBoost"] = {
            "model": XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=0),
            "use_scaler": False
        }
    if LGBMRegressor is not None:
        models["LightGBM"] = {
            "model": LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
            "use_scaler": False
        }
    if CatBoostRegressor is not None:
        models["CatBoost"] = {
            "model": CatBoostRegressor(iterations=200, depth=8, learning_rate=0.05, random_state=RANDOM_STATE, verbose=0),
            "use_scaler": False
        }

    best_name = None
    best_mae = float("inf")
    best_model = None
    best_use_scaler = False

    # ---- CV evaluation ----
    for name, cfg in models.items():
        print(f"\nEvaluating {name}...")
        mae = evaluate_model_cv(
            cfg["model"], X, y, tscv, use_scaler=cfg["use_scaler"]
        )
        print(f"{name} CV MAE: {mae:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = cfg["model"]
            best_use_scaler = cfg["use_scaler"]

    print(f"\nBest model: {best_name} (MAE: {best_mae:.3f})")

    # ---- Train final model ----
    print("\nTraining final model on full dataset...")

    if best_use_scaler:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        final_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", best_model)
        ])
    else:
        final_model = best_model

    final_model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # ---- Sanity check ----
    loaded_model = joblib.load(MODEL_PATH)
    preds = loaded_model.predict(X.iloc[:5])
    # print("Test predictions:", preds)

if __name__ == "__main__":
    main()
