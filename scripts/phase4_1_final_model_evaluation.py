from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"

TEST_FRACTION = 0.20

MODELS = [
    "Holt_Damped",
    "Holt_Linear",
    "ARIMA_1_1_1",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"Date", TARGET}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["Date", TARGET])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    if (df[TARGET] < 0).any():
        raise ValueError(
            "Negative target values found."
        )

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    nonzero = actual != 0

    if nonzero.any():
        mape = np.mean(
            np.abs(
                errors[nonzero] / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
        "Mean_Error": np.mean(errors),
        "Median_Error": np.median(errors),
    }


# ============================================================
# FIT AND FORECAST
# ============================================================

def fit_and_forecast(model_name, train, horizon):

    if model_name == "Holt_Damped":

        model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=True,
            initialization_method="estimated",
        )

        fitted = model.fit(optimized=True)

        return np.asarray(
            fitted.forecast(horizon)
        )

    if model_name == "Holt_Linear":

        model = ExponentialSmoothing(
            train,
            trend="add",
            damped_trend=False,
            initialization_method="estimated",
        )

        fitted = model.fit(optimized=True)

        return np.asarray(
            fitted.forecast(horizon)
        )

    if model_name == "ARIMA_1_1_1":

        fitted = ARIMA(
            train,
            order=(1, 1, 1),
        ).fit()

        return np.asarray(
            fitted.forecast(horizon)
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# PERSISTENCE BASELINE
# ============================================================

def naive_forecast(train, horizon):

    return np.repeat(
        train[-1],
        horizon,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_models(df):

    values = df[TARGET].to_numpy(dtype=float)

    test_size = max(
        1,
        int(np.ceil(len(df) * TEST_FRACTION)),
    )

    split_index = len(df) - test_size

    train = values[:split_index]
    actual = values[split_index:]

    test_dates = df["Date"].iloc[
        split_index:
    ].to_numpy()

    print("\n1. CHRONOLOGICAL HOLDOUT")
    print("-" * 70)

    print(f"Total observations: {len(df)}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(actual)}")

    print(
        f"Training dates: "
        f"{df['Date'].iloc[0].date()} to "
        f"{df['Date'].iloc[split_index - 1].date()}"
    )

    print(
        f"Testing dates: "
        f"{df['Date'].iloc[split_index].date()} to "
        f"{df['Date'].iloc[-1].date()}"
    )

    print(
        "\nForecast horizon is the complete holdout "
        "sequence, not calendar-day frequency."
    )

    prediction_rows = []
    metric_rows = []

    all_models = ["Naive"] + MODELS

    for model_name in all_models:

        print(f"\nEvaluating: {model_name}")

        try:

            if model_name == "Naive":
                predicted = naive_forecast(
                    train,
                    len(actual),
                )
            else:
                predicted = fit_and_forecast(
                    model_name,
                    train,
                    len(actual),
                )

            if len(predicted) != len(actual):
                raise ValueError(
                    "Prediction length does not match test length."
                )

            if not np.isfinite(predicted).all():
                raise ValueError(
                    "Non-finite predictions detected."
                )

            metrics = calculate_metrics(
                actual,
                predicted,
            )

            metric_rows.append({
                "Model": model_name,
                "Training_Observations": len(train),
                "Test_Observations": len(actual),
                **metrics,
            })

            for i in range(len(actual)):

                prediction_rows.append({
                    "Model": model_name,
                    "Date": test_dates[i],
                    "Actual": actual[i],
                    "Prediction": predicted[i],
                    "Error": actual[i] - predicted[i],
                    "Absolute_Error": abs(
                        actual[i] - predicted[i]
                    ),
                })

        except Exception as exc:

            print(
                f"WARNING: {model_name} failed: {exc}"
            )

    if not metric_rows:
        raise RuntimeError(
            "No model completed evaluation."
        )

    metrics_df = pd.DataFrame(
        metric_rows
    ).sort_values(
        ["MAE", "RMSE"]
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    predictions_df["Date"] = pd.to_datetime(
        predictions_df["Date"]
    )

    metrics_df.to_csv(
        REPORTS_DIR
        / "phase4_1_final_model_comparison.csv",
        index=False,
    )

    predictions_df.to_csv(
        REPORTS_DIR
        / "phase4_1_final_predictions.csv",
        index=False,
    )

    return metrics_df, predictions_df


# ============================================================
# PLOTS
# ============================================================

def create_plots(predictions_df):

    # --------------------------------------------------------
    # Actual versus predicted
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    actual_data = (
        predictions_df[
            ["Date", "Actual"]
        ]
        .drop_duplicates(subset=["Date"])
        .sort_values("Date")
    )

    plt.plot(
        actual_data["Date"],
        actual_data["Actual"],
        label="Actual",
        linewidth=2,
    )

    for model_name in predictions_df["Model"].unique():

        model_data = predictions_df[
            predictions_df["Model"] == model_name
        ].sort_values("Date")

        plt.plot(
            model_data["Date"],
            model_data["Prediction"],
            label=model_name,
            linewidth=1.2,
        )

    plt.title(
        "Final Holdout: Actual vs Predicted"
    )

    plt.xlabel("Observation Date")
    plt.ylabel("Children in HHS Care")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "phase4_1_final_actual_vs_predicted.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # Error comparison
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    for model_name in predictions_df["Model"].unique():

        model_data = predictions_df[
            predictions_df["Model"] == model_name
        ].sort_values("Date")

        plt.plot(
            model_data["Date"],
            model_data["Error"],
            label=model_name,
            linewidth=1.1,
        )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.title(
        "Final Holdout: Forecast Errors"
    )

    plt.xlabel("Observation Date")
    plt.ylabel("Error (Actual - Prediction)")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "phase4_1_final_error_comparison.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 4.1 - FINAL MODEL EVALUATION")
    print("=" * 70)

    df = load_data()

    metrics_df, predictions_df = evaluate_models(df)

    create_plots(predictions_df)

    print("\n2. FINAL MODEL COMPARISON")
    print("-" * 70)

    print(
        metrics_df.round(3).to_string(
            index=False
        )
    )

    print("\n3. GENERATED FILES")
    print("-" * 70)

    print(
        "reports/phase4_1_final_model_comparison.csv"
    )

    print(
        "reports/phase4_1_final_predictions.csv"
    )

    print(
        "reports/figures/phase4_1_final_actual_vs_predicted.png"
    )

    print(
        "reports/figures/phase4_1_final_error_comparison.png"
    )

    print("\nPHASE 4.1 COMPLETED")

    print(
        "\nImportant: this is a single-origin, multi-step "
        "holdout evaluation. It is not directly equivalent "
        "to the earlier seven-record walk-forward evaluation."
    )


if __name__ == "__main__":
    main()