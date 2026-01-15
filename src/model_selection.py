import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def evaluate_model_cv(model, X, y, tscv, use_scaler=False):
    """
    Manual TimeSeries CV evaluation.
    """
    maes = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_scaler:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)

    return np.mean(maes)
