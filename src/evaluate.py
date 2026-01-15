import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

MODEL_PATH = "models/final_model.joblib"
FEATURES_PATH = "data/processed/solar_features.csv"

def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)

    X = df.drop(columns=["power_generated_kw"])
    y = df["power_generated_kw"]

    # Feature importance handling for different model types
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
    else:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="neg_mean_absolute_error")
        importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importance.head(10))

    # Save to CSV (clave para Streamlit o análisis)
    importance.to_csv("models/feature_importance.csv")

    # Plot
    plt.figure(figsize=(8, 5))
    importance.head(10).plot(kind="barh", color="#7A7A7A")
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()

    features = [
        "distance_to_solar_noon",
        "sky_cover",
        "day_of_year_cos"
    ]

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features,
        grid_resolution=50
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
