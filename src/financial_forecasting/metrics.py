import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_denominator = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / safe_denominator)) * 100.0)


def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }

