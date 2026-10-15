from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    ROOT / "reports" / "phase3_4_advanced_predictions.csv"
)

MODEL_COMPARISON_FILE = (
    ROOT / "reports" / "phase3_4_advanced_model_comparison.csv"
)

HORIZON_METRICS_FILE = (
    ROOT / "reports" / "phase3_4_horizon_metrics.csv"
)

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = "Holt_Damped"

# Nominal central interval levels.
# Actual coverage must be measured, not assumed.
INTERVAL_LEVELS = [0.80, 0.90, 0.95]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    required_files = [
        PREDICTIONS_FILE,
        MODEL_COMPARISON_FILE,
        HORIZON_METRICS_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    predictions = pd.read_csv(PREDICTIONS_FILE)

    predictions["Date"] = pd.to_datetime(
        predictions["Date"],
        errors="coerce",
    )

    numeric_columns = [
        "Actual",
        "Prediction",
        "Error",
        "Forecast_Step",
        "Origin_Index",
    ]

    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

    predictions = predictions.dropna(
        subset=[
            "Date",
            "Actual",
            "Prediction",
            "Forecast_Step",
            "Origin_Index",
        ]
    ).copy()

    predictions = predictions[
        predictions["Model"] == BEST_MODEL
    ].copy()

    if predictions.empty:
        raise ValueError(
            f"No predictions found for {BEST_MODEL}"
        )

    predictions["Error"] = (
        predictions["Actual"] - predictions["Prediction"]
    )

    predictions["Absolute_Error"] = (
        predictions["Error"].abs()
    )

    predictions["Squared_Error"] = (
        predictions["Error"] ** 2
    )

    horizon_metrics = pd.read_csv(HORIZON_METRICS_FILE)

    horizon_metrics = horizon_metrics[
        horizon_metrics["Model"] == BEST_MODEL
    ].copy()

    if horizon_metrics.empty:
        raise ValueError(
            f"No horizon metrics found for {BEST_MODEL}"
        )

    return predictions, horizon_metrics


# ============================================================
# EMPIRICAL PREDICTION INTERVALS
# ============================================================

def calculate_empirical_intervals(predictions):

    """
    Calculate empirical intervals separately for each forecast step.

    Uses leave-one-origin-out calibration:
    the error from a forecast origin is excluded when estimating
    that origin's interval.

    These are empirical error intervals, not guaranteed
    distribution-free prediction intervals.
    """

    predictions = predictions.sort_values(
        ["Origin_Index", "Forecast_Step"]
    ).copy()

    rows = []

    for _, row in predictions.iterrows():

        step = int(row["Forecast_Step"])
        origin = row["Origin_Index"]

        calibration = predictions[
            (predictions["Forecast_Step"] == step)
            & (predictions["Origin_Index"] != origin)
        ]["Error"].dropna().to_numpy()

        if len(calibration) < 10:
            continue

        actual = float(row["Actual"])
        point_forecast = float(row["Prediction"])

        for level in INTERVAL_LEVELS:

            alpha = 1 - level

            # Quantiles of signed historical forecast errors.
            lower_error = np.quantile(
                calibration,
                alpha / 2,
            )

            upper_error = np.quantile(
                calibration,
                1 - alpha / 2,
            )

            # Error = Actual - Prediction.
            # Therefore prediction bounds are shifted by error quantiles.
            lower = max(
                0.0,
                point_forecast + lower_error,
            )

            upper = max(
                lower,
                point_forecast + upper_error,
            )

            rows.append({
                "Origin_Index": origin,
                "Date": row["Date"],
                "Forecast_Step": step,
                "Actual": actual,
                "Prediction": point_forecast,
                "Interval_Level": level,
                "Lower_Bound": lower,
                "Upper_Bound": upper,
                "Interval_Width": upper - lower,
                "Covered": (
                    lower <= actual <= upper
                ),
            })

    intervals = pd.DataFrame(rows)

    if intervals.empty:
        raise ValueError(
            "Could not calculate intervals. "
            "Check the number of validation origins."
        )

    intervals.to_csv(
        REPORTS_DIR
        / "phase3_5_empirical_prediction_intervals.csv",
        index=False,
    )

    return intervals


# ============================================================
# COVERAGE AND INTERVAL METRICS
# ============================================================

def calculate_coverage(intervals):

    rows = []

    for level, group in intervals.groupby("Interval_Level"):

        rows.append({
            "Nominal_Level_Percent": level * 100,
            "Forecasts_Evaluated": len(group),
            "Empirical_Coverage_Percent": (
                group["Covered"].mean() * 100
            ),
            "Average_Interval_Width": (
                group["Interval_Width"].mean()
            ),
            "Median_Interval_Width": (
                group["Interval_Width"].median()
            ),
        })

    summary = pd.DataFrame(rows).sort_values(
        "Nominal_Level_Percent"
    )

    summary.to_csv(
        REPORTS_DIR
        / "phase3_5_interval_coverage.csv",
        index=False,
    )

    return summary


def calculate_horizon_coverage(intervals):

    rows = []

    for (level, step), group in intervals.groupby(
        ["Interval_Level", "Forecast_Step"]
    ):

        rows.append({
            "Nominal_Level_Percent": level * 100,
            "Forecast_Step": step,
            "Forecasts_Evaluated": len(group),
            "Empirical_Coverage_Percent": (
                group["Covered"].mean() * 100
            ),
            "Average_Interval_Width": (
                group["Interval_Width"].mean()
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        REPORTS_DIR
        / "phase3_5_horizon_coverage.csv",
        index=False,
    )

    return result


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_plots(intervals, coverage, horizon_coverage):

    # Plot 1: Point forecast and 95% empirical interval
    selected = intervals[
        intervals["Interval_Level"] == 0.95
    ].copy()

    selected = selected.sort_values(
        ["Date", "Origin_Index"]
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        selected["Date"],
        selected["Actual"],
        label="Actual",
        linewidth=1.5,
    )

    plt.plot(
        selected["Date"],
        selected["Prediction"],
        label="Point Forecast",
        linewidth=1.2,
    )

    plt.fill_between(
        selected["Date"],
        selected["Lower_Bound"],
        selected["Upper_Bound"],
        alpha=0.2,
        label="95% Empirical Interval",
    )

    plt.title(
        "Holt Damped: Forecasts and Empirical Intervals"
    )

    plt.xlabel("Observation Date")
    plt.ylabel("Children in HHS Care")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "phase3_5_empirical_prediction_intervals.png",
        dpi=160,
    )

    plt.close()

    # Plot 2: Nominal vs empirical coverage
    plt.figure(figsize=(8, 5))

    plt.plot(
        coverage["Nominal_Level_Percent"],
        coverage["Empirical_Coverage_Percent"],
        marker="o",
        label="Observed Coverage",
    )

    plt.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        label="Ideal Calibration",
    )

    plt.title(
        "Prediction Interval Coverage"
    )

    plt.xlabel("Nominal Interval Level (%)")
    plt.ylabel("Empirical Coverage (%)")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "phase3_5_interval_coverage.png",
        dpi=160,
    )

    plt.close()

    # Plot 3: Average interval width by horizon
    width_table = horizon_coverage[
        horizon_coverage["Nominal_Level_Percent"] == 95
    ].sort_values("Forecast_Step")

    plt.figure(figsize=(9, 5))

    plt.plot(
        width_table["Forecast_Step"],
        width_table["Average_Interval_Width"],
        marker="o",
    )

    plt.title(
        "95% Empirical Interval Width by Forecast Step"
    )

    plt.xlabel("Forecast Step (Observed Record)")
    plt.ylabel("Average Interval Width")

    plt.xticks(
        width_table["Forecast_Step"]
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "phase3_5_interval_width_by_horizon.png",
        dpi=160,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 3.5 - FORECAST UNCERTAINTY")
    print("=" * 70)

    predictions, horizon_metrics = load_data()

    print(f"Model: {BEST_MODEL}")
    print(
        f"Forecasts available: {len(predictions)}"
    )

    print(
        f"Forecast origins: "
        f"{predictions['Origin_Index'].nunique()}"
    )

    intervals = calculate_empirical_intervals(
        predictions
    )

    coverage = calculate_coverage(
        intervals
    )

    horizon_coverage = calculate_horizon_coverage(
        intervals
    )

    print("\n1. INTERVAL COVERAGE")
    print("-" * 70)

    print(
        coverage.round(3).to_string(index=False)
    )

    print("\n2. COVERAGE BY FORECAST STEP")
    print("-" * 70)

    print(
        horizon_coverage.round(3).to_string(index=False)
    )

    create_plots(
        intervals,
        coverage,
        horizon_coverage,
    )

    print("\n3. GENERATED FILES")
    print("-" * 70)

    print(
        "reports/phase3_5_empirical_prediction_intervals.csv"
    )

    print(
        "reports/phase3_5_interval_coverage.csv"
    )

    print(
        "reports/phase3_5_horizon_coverage.csv"
    )

    print(
        "reports/figures/phase3_5_empirical_prediction_intervals.png"
    )

    print(
        "reports/figures/phase3_5_interval_coverage.png"
    )

    print(
        "reports/figures/phase3_5_interval_width_by_horizon.png"
    )

    print("\nPHASE 3.5 COMPLETED")

    print(
        "\nCaution: empirical coverage is measured on historical "
        "walk-forward predictions. Nominal 80%, 90%, and 95% "
        "levels are targets, not guaranteed coverage."
    )

    print(
        "Forecast steps refer to observed records, not calendar days."
    )


if __name__ == "__main__":
    main()