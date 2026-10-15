from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "phase3_4_advanced_predictions.csv"
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = "Holt_Damped"

INTERVAL_LEVELS = [0.80, 0.90, 0.95]

# First 60% of chronological forecast origins are used
# for calibration; remaining origins are evaluation data.
CALIBRATION_FRACTION = 0.60

MIN_CALIBRATION_ORIGINS = 10


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "Model",
        "Origin_Index",
        "Date",
        "Forecast_Step",
        "Actual",
        "Prediction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df[df["Model"] == BEST_MODEL].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    for column in [
        "Origin_Index",
        "Forecast_Step",
        "Actual",
        "Prediction",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Date",
            "Origin_Index",
            "Forecast_Step",
            "Actual",
            "Prediction",
        ]
    ).copy()

    df["Origin_Index"] = df["Origin_Index"].astype(int)
    df["Forecast_Step"] = df["Forecast_Step"].astype(int)

    df["Error"] = df["Actual"] - df["Prediction"]

    df = df.sort_values(
        ["Origin_Index", "Forecast_Step"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No predictions available for {BEST_MODEL}"
        )

    return df


# ============================================================
# CHRONOLOGICAL CALIBRATION / EVALUATION SPLIT
# ============================================================

def split_origins(df):

    origins = np.sort(
        df["Origin_Index"].unique()
    )

    split_position = int(
        len(origins) * CALIBRATION_FRACTION
    )

    split_position = max(
        split_position,
        MIN_CALIBRATION_ORIGINS,
    )

    if split_position >= len(origins):
        raise ValueError(
            "Not enough origins left for a separate evaluation set."
        )

    calibration_origins = origins[:split_position]
    evaluation_origins = origins[split_position:]

    calibration = df[
        df["Origin_Index"].isin(calibration_origins)
    ].copy()

    evaluation = df[
        df["Origin_Index"].isin(evaluation_origins)
    ].copy()

    print(f"Total forecast origins: {len(origins)}")
    print(
        f"Calibration origins: {len(calibration_origins)}"
    )
    print(
        f"Evaluation origins: {len(evaluation_origins)}"
    )

    print(
        f"Last calibration origin index: "
        f"{calibration_origins[-1]}"
    )

    print(
        f"First evaluation origin index: "
        f"{evaluation_origins[0]}"
    )

    return calibration, evaluation


# ============================================================
# HORIZON-SPECIFIC ABSOLUTE ERROR QUANTILES
# ============================================================

def calculate_calibration_thresholds(calibration):

    rows = []

    for step, group in calibration.groupby("Forecast_Step"):

        absolute_errors = np.abs(
            group["Error"].to_numpy()
        )

        if len(absolute_errors) < MIN_CALIBRATION_ORIGINS:
            raise ValueError(
                f"Insufficient calibration errors for step {step}"
            )

        for level in INTERVAL_LEVELS:

            # Conservative finite-sample quantile rank.
            n = len(absolute_errors)

            rank = int(
                np.ceil((n + 1) * level)
            )

            if rank > n:
                threshold = np.inf
            else:
                threshold = np.sort(
                    absolute_errors
                )[rank - 1]

            rows.append({
                "Forecast_Step": int(step),
                "Nominal_Level_Percent": level * 100,
                "Calibration_Count": n,
                "Absolute_Error_Threshold": threshold,
            })

    thresholds = pd.DataFrame(rows)

    thresholds.to_csv(
        REPORTS_DIR
        / "phase3_6_calibration_thresholds.csv",
        index=False,
    )

    return thresholds


# ============================================================
# APPLY INTERVALS TO EVALUATION DATA
# ============================================================

def construct_evaluation_intervals(
    evaluation,
    thresholds,
):

    rows = []

    for _, prediction in evaluation.iterrows():

        step = int(
            prediction["Forecast_Step"]
        )

        point = float(
            prediction["Prediction"]
        )

        actual = float(
            prediction["Actual"]
        )

        for level in INTERVAL_LEVELS:

            match = thresholds[
                (thresholds["Forecast_Step"] == step)
                & (
                    thresholds["Nominal_Level_Percent"]
                    == level * 100
                )
            ]

            if match.empty:
                continue

            radius = float(
                match["Absolute_Error_Threshold"].iloc[0]
            )

            lower = max(
                0.0,
                point - radius,
            )

            upper = point + radius

            rows.append({
                "Origin_Index": int(
                    prediction["Origin_Index"]
                ),
                "Date": prediction["Date"],
                "Forecast_Step": step,
                "Actual": actual,
                "Prediction": point,
                "Nominal_Level_Percent": level * 100,
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
            "No evaluation intervals were constructed."
        )

    intervals.to_csv(
        REPORTS_DIR
        / "phase3_6_calibrated_evaluation_intervals.csv",
        index=False,
    )

    return intervals


# ============================================================
# COVERAGE SUMMARY
# ============================================================

def calculate_coverage(intervals):

    rows = []

    for level, group in intervals.groupby(
        "Nominal_Level_Percent"
    ):

        rows.append({
            "Nominal_Level_Percent": level,
            "Evaluation_Count": len(group),
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
        / "phase3_6_calibration_coverage.csv",
        index=False,
    )

    return summary


def calculate_horizon_coverage(intervals):

    rows = []

    for (level, step), group in intervals.groupby(
        [
            "Nominal_Level_Percent",
            "Forecast_Step",
        ]
    ):

        rows.append({
            "Nominal_Level_Percent": level,
            "Forecast_Step": step,
            "Evaluation_Count": len(group),
            "Empirical_Coverage_Percent": (
                group["Covered"].mean() * 100
            ),
            "Average_Interval_Width": (
                group["Interval_Width"].mean()
            ),
        })

    result = pd.DataFrame(rows).sort_values(
        [
            "Nominal_Level_Percent",
            "Forecast_Step",
        ]
    )

    result.to_csv(
        REPORTS_DIR
        / "phase3_6_horizon_coverage.csv",
        index=False,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 3.6 - CHRONOLOGICAL INTERVAL CALIBRATION")
    print("=" * 70)

    predictions = load_predictions()

    print(f"Model: {BEST_MODEL}")
    print(f"Available forecasts: {len(predictions)}")

    calibration, evaluation = split_origins(
        predictions
    )

    print("\n1. CALIBRATION THRESHOLDS")
    print("-" * 70)

    thresholds = calculate_calibration_thresholds(
        calibration
    )

    print(
        thresholds.round(3).to_string(index=False)
    )

    intervals = construct_evaluation_intervals(
        evaluation,
        thresholds,
    )

    coverage = calculate_coverage(
        intervals
    )

    horizon_coverage = calculate_horizon_coverage(
        intervals
    )

    print("\n2. HELD-OUT EVALUATION COVERAGE")
    print("-" * 70)

    print(
        coverage.round(3).to_string(index=False)
    )

    print("\n3. COVERAGE BY FORECAST STEP")
    print("-" * 70)

    print(
        horizon_coverage.round(3).to_string(index=False)
    )

    print("\n4. GENERATED FILES")
    print("-" * 70)

    print(
        "reports/phase3_6_calibration_thresholds.csv"
    )

    print(
        "reports/phase3_6_calibrated_evaluation_intervals.csv"
    )

    print(
        "reports/phase3_6_calibration_coverage.csv"
    )

    print(
        "reports/phase3_6_horizon_coverage.csv"
    )

    print("\nPHASE 3.6 COMPLETED")

    print(
        "\nNote: These empirical intervals are not guaranteed "
        "to attain their nominal coverage."
    )

    print(
        "The chronological holdout is small, and overlapping "
        "forecast windows can reduce the reliability of coverage "
        "estimates."
    )

    print(
        "Forecast steps represent observed records, not calendar days."
    )


if __name__ == "__main__":
    main()