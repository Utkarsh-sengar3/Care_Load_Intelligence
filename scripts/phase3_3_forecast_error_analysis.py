from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    ROOT / "reports" / "phase3_2_walk_forward_predictions.csv"
)

HORIZON_FILE = (
    ROOT / "reports" / "phase3_2_horizon_metrics.csv"
)

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = "Exponential_Smoothing"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    if not HORIZON_FILE.exists():
        raise FileNotFoundError(
            f"Horizon metrics file not found: {HORIZON_FILE}"
        )

    predictions = pd.read_csv(PREDICTIONS_FILE)
    horizon_metrics = pd.read_csv(HORIZON_FILE)

    required_columns = {
        "Model",
        "Date",
        "Actual",
        "Prediction",
        "Forecast_Step",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing columns in predictions file: {missing}"
        )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"],
        errors="coerce",
    )

    predictions = predictions.dropna(
        subset=["Date", "Actual", "Prediction"]
    ).copy()

    predictions["Error"] = (
        predictions["Actual"] - predictions["Prediction"]
    )

    predictions["Absolute_Error"] = (
        predictions["Error"].abs()
    )

    predictions["Squared_Error"] = (
        predictions["Error"] ** 2
    )

    predictions["Percentage_Error"] = np.where(
        predictions["Actual"] != 0,
        predictions["Error"]
        / predictions["Actual"]
        * 100,
        np.nan,
    )

    return predictions, horizon_metrics


# ============================================================
# ERROR SUMMARY
# ============================================================

def calculate_error_summary(predictions):

    rows = []

    for model, group in predictions.groupby("Model"):

        actual = group["Actual"].to_numpy()
        predicted = group["Prediction"].to_numpy()
        errors = group["Error"].to_numpy()

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        mape = np.mean(
            np.abs(
                (actual[actual != 0]
                 - predicted[actual != 0])
                / actual[actual != 0]
            )
        ) * 100

        rows.append({
            "Model": model,
            "Forecasts_Evaluated": len(group),
            "MAE": mae,
            "RMSE": rmse,
            "Mean_Error": np.mean(errors),
            "Median_Error": np.median(errors),
            "MAPE_Percent": mape,
            "Error_Std": np.std(errors),
            "Overprediction_Count": int(
                np.sum(predicted > actual)
            ),
            "Underprediction_Count": int(
                np.sum(predicted < actual)
            ),
            "Exact_Prediction_Count": int(
                np.sum(predicted == actual)
            ),
        })

    summary = pd.DataFrame(rows).sort_values("MAE")

    summary.to_csv(
        REPORTS_DIR / "phase3_3_error_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# BEST MODEL DIAGNOSTICS
# ============================================================

def create_diagnostic_plots(predictions, horizon_metrics):

    best = predictions[
        predictions["Model"] == BEST_MODEL
    ].copy()

    if best.empty:
        raise ValueError(
            f"No predictions found for model: {BEST_MODEL}"
        )

    best = best.sort_values(
    ["Date", "Forecast_Step"]
    )

    # --------------------------------------------------------
    # 1. Actual vs Predicted
    # --------------------------------------------------------

    plt.figure(figsize=(14, 6))

    plt.plot(
        best["Date"],
        best["Actual"],
        label="Actual",
        linewidth=2,
    )

    plt.plot(
        best["Date"],
        best["Prediction"],
        label="Predicted",
        linewidth=1.5,
    )

    plt.title(
        "Walk-Forward Forecast: Actual vs Predicted"
    )

    plt.xlabel("Observation Date")
    plt.ylabel("Children in HHS Care")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase3_3_actual_vs_predicted.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # 2. Forecast Error Over Time
    # --------------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(
        best["Date"],
        best["Error"],
        linewidth=1,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.title(
        "Forecast Error Over Time "
        "(Actual - Prediction)"
    )

    plt.xlabel("Observation Date")
    plt.ylabel("Forecast Error")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase3_3_error_over_time.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Error Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.hist(
        best["Error"],
        bins=25,
        edgecolor="black",
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.title(
        "Distribution of Forecast Errors"
    )

    plt.xlabel("Error (Actual - Prediction)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase3_3_error_distribution.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # 4. Actual vs Predicted Scatter
    # --------------------------------------------------------

    plt.figure(figsize=(7, 7))

    plt.scatter(
        best["Actual"],
        best["Prediction"],
        alpha=0.65,
    )

    lower = min(
        best["Actual"].min(),
        best["Prediction"].min(),
    )

    upper = max(
        best["Actual"].max(),
        best["Prediction"].max(),
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )

    plt.title(
        "Actual vs Predicted Values"
    )

    plt.xlabel("Actual HHS Care Load")
    plt.ylabel("Predicted HHS Care Load")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase3_3_actual_predicted_scatter.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # 5. Horizon-Wise MAE
    # --------------------------------------------------------

    best_horizon = horizon_metrics[
        horizon_metrics["Model"] == BEST_MODEL
    ].sort_values("Forecast_Step")

    plt.figure(figsize=(9, 5))

    plt.plot(
        best_horizon["Forecast_Step"],
        best_horizon["MAE"],
        marker="o",
    )

    plt.title(
        "Forecast Error by Horizon"
    )

    plt.xlabel("Forecast Step (Observed Record)")
    plt.ylabel("MAE")

    plt.xticks(
        best_horizon["Forecast_Step"]
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase3_3_horizon_mae.png",
        dpi=160,
    )

    plt.close()

    # --------------------------------------------------------
    # 6. Save Detailed Predictions
    # --------------------------------------------------------

    best.to_csv(
        REPORTS_DIR / "phase3_3_best_model_predictions.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 3.3 - FORECAST ERROR ANALYSIS")
    print("=" * 70)

    predictions, horizon_metrics = load_data()

    print("\n1. ERROR SUMMARY")
    print("-" * 70)

    summary = calculate_error_summary(predictions)

    print(
        summary.round(3).to_string(index=False)
    )

    print("\n2. BEST MODEL")
    print("-" * 70)

    print(f"Selected diagnostic model: {BEST_MODEL}")

    create_diagnostic_plots(
        predictions,
        horizon_metrics,
    )

    print("\n3. GENERATED REPORTS")
    print("-" * 70)

    print(
        "reports/phase3_3_error_summary.csv"
    )

    print(
        "reports/phase3_3_best_model_predictions.csv"
    )

    print(
        "reports/figures/phase3_3_actual_vs_predicted.png"
    )

    print(
        "reports/figures/phase3_3_error_over_time.png"
    )

    print(
        "reports/figures/phase3_3_error_distribution.png"
    )

    print(
        "reports/figures/phase3_3_actual_predicted_scatter.png"
    )

    print(
        "reports/figures/phase3_3_horizon_mae.png"
    )

    print("\nPHASE 3.3 COMPLETED")


if __name__ == "__main__":
    main()