import numpy as np
from sklearn.metrics import mean_absolute_error

def naive_last_value_baseline(y):
    """
    Naive baseline: predict previous timestep value.
    """
    y_true = y.iloc[1:]
    y_pred = y.shift(1).iloc[1:]
    mae = mean_absolute_error(y_true, y_pred)
    return mae
