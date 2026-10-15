from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "uac_observed_forecasting_data.csv"
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"

# Minimum number of observations before validation begins.
INITIAL_TRAIN_SIZE = 400

# Evaluate the next 7 observed records at each origin.
FORECAST_HORIZON = 7

# Advance the forecast origin by 7 observed records.
STEP_SIZE = 7


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
    }


def main():
    print("\n" + "=" * 70)
    print("PHASE 3.2 - WALK-FORWARD VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    ).sort_values("Date").reset_index(drop=True)

    values = df[TARGET].astype(float).reset_index(drop=True)
    dates = df["Date"].reset_index(drop=True)

    if len(values) <= INITIAL_TRAIN_SIZE + FORECAST_HORIZON:
        raise ValueError(
            "Not enough observations for walk-forward validation."
        )

    all_predictions = []
    model_errors = []

    origins = range(
        INITIAL_TRAIN_SIZE,
        len(values) - FORECAST_HORIZON + 1,
        STEP_SIZE,
    )

    print(f"Total observations: {len(values)}")
    print(f"Initial training size: {INITIAL_TRAIN_SIZE}")
    print(f"Forecast horizon: {FORECAST_HORIZON} observed records")
    print(f"Step size: {STEP_SIZE} observed records")

    for origin in origins:
        train = values.iloc[:origin]
        actual = values.iloc[
            origin:origin + FORECAST_HORIZON
        ]

        forecast_dates = dates.iloc[
            origin:origin + FORECAST_HORIZON
        ]

        # --------------------------------------------------
        # Naive baseline
        # --------------------------------------------------
        naive_forecast = np.repeat(
            train.iloc[-1],
            len(actual),
        )

        all_predictions.append({
            "Origin_Index": origin,
            "Model": "Naive",
            "Forecast_Step": np.arange(1, len(actual) + 1),
            "Date": forecast_dates.to_numpy(),
            "Actual": actual.to_numpy(),
            "Prediction": naive_forecast,
        })

        # --------------------------------------------------
        # Exponential Smoothing
        # --------------------------------------------------
        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                damped_trend=True,
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)

            forecast = model.forecast(len(actual)).to_numpy()

            all_predictions.append({
                "Origin_Index": origin,
                "Model": "Exponential_Smoothing",
                "Forecast_Step": np.arange(1, len(actual) + 1),
                "Date": forecast_dates.to_numpy(),
                "Actual": actual.to_numpy(),
                "Prediction": forecast,
            })

        except Exception as exc:
            model_errors.append({
                "Origin_Index": origin,
                "Model": "Exponential_Smoothing",
                "Error": str(exc),
            })

        # --------------------------------------------------
        # ARIMA
        # --------------------------------------------------
        try:
            model = ARIMA(
                train,
                order=(1, 1, 1),
            ).fit()

            forecast = model.forecast(len(actual)).to_numpy()

            all_predictions.append({
                "Origin_Index": origin,
                "Model": "ARIMA_1_1_1",
                "Forecast_Step": np.arange(1, len(actual) + 1),
                "Date": forecast_dates.to_numpy(),
                "Actual": actual.to_numpy(),
                "Prediction": forecast,
            })

        except Exception as exc:
            model_errors.append({
                "Origin_Index": origin,
                "Model": "ARIMA_1_1_1",
                "Error": str(exc),
            })

    # --------------------------------------------------
    # Combine predictions
    # --------------------------------------------------
    if not all_predictions:
        raise RuntimeError(
            "No predictions were generated."
        )

    predictions_df = pd.concat(
        [
            pd.DataFrame({
                "Origin_Index": item["Origin_Index"],
                "Model": item["Model"],
                "Forecast_Step": item["Forecast_Step"],
                "Date": item["Date"],
                "Actual": item["Actual"],
                "Prediction": item["Prediction"],
            })
            for item in all_predictions
        ],
        ignore_index=True,
    )

    predictions_df["Absolute_Error"] = (
        predictions_df["Actual"]
        - predictions_df["Prediction"]
    ).abs()

    predictions_df["Squared_Error"] = (
        predictions_df["Actual"]
        - predictions_df["Prediction"]
    ) ** 2

    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------
    summary_rows = []

    for model_name, group in predictions_df.groupby("Model"):
        metrics = calculate_metrics(
            group["Actual"],
            group["Prediction"],
        )

        summary_rows.append({
            "Model": model_name,
            "Forecasts_Evaluated": len(group),
            "Forecast_Origins": group["Origin_Index"].nunique(),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "MAE"
    )

    summary_df.to_csv(
        REPORTS_DIR / "phase3_2_walk_forward_summary.csv",
        index=False,
    )

    predictions_df.to_csv(
        REPORTS_DIR / "phase3_2_walk_forward_predictions.csv",
        index=False,
    )

    if model_errors:
        pd.DataFrame(model_errors).to_csv(
            REPORTS_DIR / "phase3_2_model_errors.csv",
            index=False,
        )

    # --------------------------------------------------
    # Horizon-wise metrics
    # --------------------------------------------------
    horizon_rows = []

    for (model_name, horizon), group in predictions_df.groupby(
        ["Model", "Forecast_Step"]
    ):
        metrics = calculate_metrics(
            group["Actual"],
            group["Prediction"],
        )

        horizon_rows.append({
            "Model": model_name,
            "Forecast_Step": horizon,
            "Forecasts_Evaluated": len(group),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
        })

    horizon_df = pd.DataFrame(horizon_rows)

    horizon_df.to_csv(
        REPORTS_DIR / "phase3_2_horizon_metrics.csv",
        index=False,
    )

    print("\n1. WALK-FORWARD MODEL COMPARISON")
    print("-" * 70)
    print(summary_df.round(3).to_string(index=False))

    print("\n2. HORIZON-WISE METRICS")
    print("-" * 70)
    print(horizon_df.round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("PHASE 3.2 COMPLETED")
    print("=" * 70)

    print("reports/phase3_2_walk_forward_summary.csv")
    print("reports/phase3_2_walk_forward_predictions.csv")
    print("reports/phase3_2_horizon_metrics.csv")

    if model_errors:
        print(
            "Some model fits failed. Details saved to "
            "reports/phase3_2_model_errors.csv"
        )

    print(
        "\nForecast horizon means the next seven observed records, "
        "not seven calendar days."
    )


if __name__ == "__main__":
    main()