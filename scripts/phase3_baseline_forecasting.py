from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

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
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"
TRAIN_RATIO = 0.80


def calculate_metrics(actual, predicted):
    """Calculate forecasting evaluation metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "MAPE_Percent": (
            np.mean(
                np.abs(
                    (actual - predicted) / actual
                )
            )
            * 100
            if np.all(actual != 0)
            else np.nan
        ),
    }


def main():
    print("\n" + "=" * 70)
    print("PHASE 3.1 - BASELINE FORECASTING")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values("Date").reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    values = df[TARGET].astype(float).reset_index(drop=True)
    dates = df["Date"].reset_index(drop=True)

    # --------------------------------------------------
    # 2. Chronological train-test split
    # --------------------------------------------------
    split_index = int(len(df) * TRAIN_RATIO)

    train = values.iloc[:split_index]
    test = values.iloc[split_index:]

    test_dates = dates.iloc[split_index:]

    if len(train) < 20 or len(test) < 1:
        raise ValueError(
            "Not enough observations for the selected split."
        )

    print("\n1. DATASET AND SPLIT")
    print("-" * 70)

    print(f"Total observations: {len(df)}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")

    print(f"Training start: {dates.iloc[0].date()}")
    print(
        "Training end:",
        dates.iloc[split_index - 1].date(),
    )
    print(f"Testing start: {test_dates.iloc[0].date()}")
    print(f"Testing end: {test_dates.iloc[-1].date()}")

    print(
        "\nForecast interpretation: "
        "next observed record, not next calendar day."
    )

    # --------------------------------------------------
    # 3. Naive forecast
    # --------------------------------------------------
    naive_predictions = np.repeat(
        train.iloc[-1],
        len(test),
    )

    # --------------------------------------------------
    # 4. Drift forecast
    # --------------------------------------------------
    drift_per_step = (
        (train.iloc[-1] - train.iloc[0])
        / (len(train) - 1)
    )

    drift_predictions = (
        train.iloc[-1]
        + drift_per_step
        * np.arange(1, len(test) + 1)
    )

    # --------------------------------------------------
    # 5. Exponential Smoothing
    # --------------------------------------------------
    predictions = {}
    errors = []

    predictions["Naive"] = naive_predictions
    predictions["Drift"] = drift_predictions

    try:
        ets_model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)

        predictions["Exponential_Smoothing"] = (
            ets_model.forecast(len(test)).to_numpy()
        )

    except Exception as exc:
        errors.append({
            "Model": "Exponential_Smoothing",
            "Error": str(exc),
        })

    # --------------------------------------------------
    # 6. ARIMA
    # --------------------------------------------------
    try:
        arima_model = ARIMA(
            train,
            order=(1, 1, 1),
        ).fit()

        predictions["ARIMA_1_1_1"] = (
            arima_model.forecast(len(test)).to_numpy()
        )

    except Exception as exc:
        errors.append({
            "Model": "ARIMA_1_1_1",
            "Error": str(exc),
        })

    # --------------------------------------------------
    # 7. Evaluate models
    # --------------------------------------------------
    metric_rows = []

    for model_name, forecast in predictions.items():
        forecast = np.asarray(forecast, dtype=float)

        if len(forecast) != len(test):
            errors.append({
                "Model": model_name,
                "Error": "Prediction length mismatch",
            })
            continue

        if not np.isfinite(forecast).all():
            errors.append({
                "Model": model_name,
                "Error": "Non-finite forecast values",
            })
            continue

        metrics = calculate_metrics(
            test.to_numpy(),
            forecast,
        )

        metric_rows.append({
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE_Percent": metrics["MAPE_Percent"],
        })

    metrics_df = pd.DataFrame(metric_rows)

    if metrics_df.empty:
        raise RuntimeError(
            "No models completed successfully."
        )

    metrics_df = metrics_df.sort_values(
        "MAE"
    ).reset_index(drop=True)

    metrics_path = (
        REPORTS_DIR
        / "phase3_baseline_model_comparison.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)

    # --------------------------------------------------
    # 8. Save test predictions
    # --------------------------------------------------
    forecast_df = pd.DataFrame({
        "Date": test_dates,
        "Actual": test.to_numpy(),
    })

    for model_name, forecast in predictions.items():
        if (
            len(forecast) == len(test)
            and np.isfinite(forecast).all()
        ):
            forecast_df[f"{model_name}_Prediction"] = forecast

    forecast_path = (
        REPORTS_DIR
        / "phase3_baseline_predictions.csv"
    )

    forecast_df.to_csv(
        forecast_path,
        index=False,
    )

    # --------------------------------------------------
    # 9. Plot actual vs predicted values
    # --------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 7))

    plt.plot(
        dates.iloc[:split_index],
        train,
        label="Training observations",
        linewidth=1.2,
    )

    plt.plot(
        test_dates,
        test,
        label="Actual test observations",
        linewidth=1.8,
    )

    for model_name, forecast in predictions.items():
        if (
            len(forecast) == len(test)
            and np.isfinite(forecast).all()
        ):
            plt.plot(
                test_dates,
                forecast,
                label=model_name,
                alpha=0.85,
            )

    plt.axvline(
        test_dates.iloc[0],
        linestyle="--",
        label="Train/test boundary",
    )

    plt.title("Baseline Forecasting: HHS Care Load")
    plt.xlabel("Observation Date")
    plt.ylabel("Children in HHS Care")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase3_baseline_forecasts.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 10. Print results
    # --------------------------------------------------
    print("\n2. MODEL COMPARISON")
    print("-" * 70)

    print(metrics_df.round(3).to_string(index=False))

    print("\n3. FORECAST SAMPLE")
    print("-" * 70)

    print(forecast_df.head(10).round(2).to_string(index=False))

    if errors:
        print("\n4. MODEL WARNINGS")
        print("-" * 70)

        for error in errors:
            print(f"{error['Model']}: {error['Error']}")

    print("\n" + "=" * 70)
    print("PHASE 3.1 COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("reports/phase3_baseline_model_comparison.csv")
    print("reports/phase3_baseline_predictions.csv")
    print("reports/figures/phase3_baseline_forecasts.png")

    print(
        "\nImportant: This is an initial benchmark over "
        "the observed-record sequence."
    )
    print(
        "Because observation dates are irregular, these "
        "results do not establish calendar-day forecasting accuracy."
    )


if __name__ == "__main__":
    main()