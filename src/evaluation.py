import numpy as np

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty set.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require finite values.")
    error = y_true - y_pred
    nonzero = y_true != 0
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_percent_nonzero_actuals": float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None,
        "forecast_bias_pred_minus_actual": float(np.mean(y_pred - y_true)),
    }
