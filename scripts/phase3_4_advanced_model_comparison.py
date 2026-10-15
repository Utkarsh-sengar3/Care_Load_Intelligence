from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "uac_observed_forecasting_data.csv"
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"

INITIAL_TRAIN_SIZE = 400
FORECAST_HORIZON = 7
STEP_SIZE = 7


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["Date", TARGET])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    values = df[TARGET].astype(float).to_numpy()

    if len(values) <= INITIAL_TRAIN_SIZE + FORECAST_HORIZON:
        raise ValueError(
            "Insufficient observations for the configured split."
        )

    return df, values


# ============================================================
# FORECASTING MODELS
# ============================================================

def forecast_naive(train, horizon):
    return np.repeat(train[-1], horizon)


def forecast_drift(train, horizon):

    if len(train) < 2:
        return forecast_naive(train, horizon)

    drift = (train[-1] - train[0]) / (len(train) - 1)

    return train[-1] + drift * np.arange(1, horizon + 1)


def forecast_holt_linear(train, horizon):

    model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=False,
        initialization_method="estimated",
    )

    fitted = model.fit(optimized=True)

    return np.asarray(fitted.forecast(horizon))


def forecast_holt_damped(train, horizon):

    model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    )

    fitted = model.fit(optimized=True)

    return np.asarray(fitted.forecast(horizon))


def forecast_ets_no_trend(train, horizon):

    model = ExponentialSmoothing(
        train,
        trend=None,
        initialization_method="estimated",
    )

    fitted = model.fit(optimized=True)

    return np.asarray(fitted.forecast(horizon))


def forecast_arima_101(train, horizon):

    fitted = ARIMA(
        train,
        order=(1, 0, 1),
    ).fit()

    return np.asarray(fitted.forecast(horizon))


def forecast_arima_111(train, horizon):

    fitted = ARIMA(
        train,
        order=(1, 1, 1),
    ).fit()

    return np.asarray(fitted.forecast(horizon))


def forecast_arima_211(train, horizon):

    fitted = ARIMA(
        train,
        order=(2, 1, 1),
    ).fit()

    return np.asarray(fitted.forecast(horizon))


MODELS = {
    "Naive": forecast_naive,
    "Drift": forecast_drift,
    "Holt_Linear": forecast_holt_linear,
    "Holt_Damped": forecast_holt_damped,
    "ETS_No_Trend": forecast_ets_no_trend,
    "ARIMA_1_0_1": forecast_arima_101,
    "ARIMA_1_1_1": forecast_arima_111,
    "ARIMA_2_1_1": forecast_arima_211,
}


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def evaluate_models(df, values):

    all_predictions = []
    model_errors = []

    origins = range(
        INITIAL_TRAIN_SIZE,
        len(values) - FORECAST_HORIZON + 1,
        STEP_SIZE,
    )

    total_origins = len(origins)

    print(f"Forecast origins: {total_origins}")
    print(f"Models: {len(MODELS)}")

    for origin_number, origin in enumerate(origins, start=1):

        train = values[:origin]
        actual = values[origin:origin + FORECAST_HORIZON]

        dates = df["Date"].iloc[
            origin:origin + FORECAST_HORIZON
        ].to_numpy()

        for model_name, forecast_function in MODELS.items():

            try:
                predicted = forecast_function(
                    train,
                    FORECAST_HORIZON,
                )

                if len(predicted) != len(actual):
                    raise ValueError(
                        "Forecast length does not match actual length."
                    )

                if not np.isfinite(predicted).all():
                    raise ValueError(
                        "Model generated non-finite predictions."
                    )

                for step in range(FORECAST_HORIZON):

                    error = (
                        actual[step] - predicted[step]
                    )

                    all_predictions.append({
                        "Model": model_name,
                        "Origin_Index": origin,
                        "Date": dates[step],
                        "Forecast_Step": step + 1,
                        "Actual": float(actual[step]),
                        "Prediction": float(predicted[step]),
                        "Error": float(error),
                        "Absolute_Error": float(abs(error)),
                        "Squared_Error": float(error ** 2),
                    })

            except Exception as exc:

                model_errors.append({
                    "Origin_Index": origin,
                    "Model": model_name,
                    "Error_Message": str(exc),
                })

        if origin_number % 5 == 0 or origin_number == total_origins:
            print(
                f"Completed origin "
                f"{origin_number}/{total_origins}"
            )

    predictions = pd.DataFrame(all_predictions)

    if predictions.empty:
        raise RuntimeError(
            "No forecasts were generated. Check the model errors."
        )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    predictions.to_csv(
        REPORTS_DIR
        / "phase3_4_advanced_predictions.csv",
        index=False,
    )

    if model_errors:

        pd.DataFrame(model_errors).to_csv(
            REPORTS_DIR
            / "phase3_4_model_errors.csv",
            index=False,
        )

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predictions):

    rows = []

    for model_name, group in predictions.groupby("Model"):

        errors = group["Error"].to_numpy()
        actual = group["Actual"].to_numpy()

        rows.append({
            "Model": model_name,
            "Forecasts_Evaluated": len(group),
            "Forecast_Origins": group["Origin_Index"].nunique(),
            "MAE": np.mean(np.abs(errors)),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
            "MAPE_Percent": np.mean(
                np.abs(errors / actual)
            ) * 100,
            "Mean_Error": np.mean(errors),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["MAE", "RMSE"]
    )

    summary.to_csv(
        REPORTS_DIR
        / "phase3_4_advanced_model_comparison.csv",
        index=False,
    )

    horizon_rows = []

    for (model_name, step), group in predictions.groupby(
        ["Model", "Forecast_Step"]
    ):

        errors = group["Error"].to_numpy()

        horizon_rows.append({
            "Model": model_name,
            "Forecast_Step": step,
            "Forecasts_Evaluated": len(group),
            "MAE": np.mean(np.abs(errors)),
            "RMSE": np.sqrt(np.mean(errors ** 2)),
        })

    horizon_metrics = pd.DataFrame(horizon_rows)

    horizon_metrics.to_csv(
        REPORTS_DIR
        / "phase3_4_horizon_metrics.csv",
        index=False,
    )

    return summary, horizon_metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 3.4 - ADVANCED MODEL COMPARISON")
    print("=" * 70)

    df, values = load_data()

    print(f"Observations: {len(df)}")
    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date: {df['Date'].max().date()}")

    print(f"Initial training size: {INITIAL_TRAIN_SIZE}")
    print(f"Forecast horizon: {FORECAST_HORIZON} observed records")
    print(f"Step size: {STEP_SIZE} observed records")

    predictions = evaluate_models(df, values)

    summary, horizon_metrics = calculate_metrics(
        predictions
    )

    print("\n1. ADVANCED MODEL COMPARISON")
    print("-" * 70)

    print(
        summary.round(3).to_string(index=False)
    )

    print("\n2. HORIZON-WISE MAE")
    print("-" * 70)

    horizon_table = horizon_metrics.pivot(
        index="Forecast_Step",
        columns="Model",
        values="MAE",
    )

    print(horizon_table.round(2).to_string())

    print("\n3. OUTPUT FILES")
    print("-" * 70)

    print(
        "reports/phase3_4_advanced_model_comparison.csv"
    )

    print(
        "reports/phase3_4_advanced_predictions.csv"
    )

    print(
        "reports/phase3_4_horizon_metrics.csv"
    )

    print("\nPHASE 3.4 COMPLETED")

    print(
        "\nInterpretation: forecasts cover the next seven observed "
        "records, not seven calendar days."
    )


if __name__ == "__main__":
    main()