from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing


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

MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"


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

    if len(df) < 50:
        raise ValueError(
            "Not enough observations to fit the final model."
        )

    values = df[TARGET].astype(float).to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite target values detected."
        )

    if (values < 0).any():
        raise ValueError(
            "Negative care-load values detected."
        )

    return df, values


# ============================================================
# TRAIN AND SAVE
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("PHASE 4.2 - FINAL MODEL ARTIFACT")
    print("=" * 70)

    df, values = load_data()

    print(f"Training observations: {len(values)}")
    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date: {df['Date'].max().date()}")

    # Fit Holt Damped on the complete available dataset.
    model = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    )

    fitted_model = model.fit(
        optimized=True
    )

    model_path = (
        MODEL_DIR
        / "holt_damped_care_load.joblib"
    )

    joblib.dump(
        fitted_model,
        model_path,
    )

    # Save benchmark information.
    metadata = {
        "target": TARGET,
        "selected_candidate": "Holt_Damped",
        "benchmark_model": "Naive",
        "training_observations": int(len(values)),
        "first_observation_date": str(
            df["Date"].min().date()
        ),
        "last_observation_date": str(
            df["Date"].max().date()
        ),
        "latest_observed_value": float(values[-1]),
        "forecast_frequency": (
            "Observed-record sequence; "
            "not a validated calendar-day forecast"
        ),
        "walk_forward_mae": 78.920,
        "walk_forward_rmse": 160.868,
        "walk_forward_mape_percent": 1.965,
        "final_holdout_mae": 211.907,
        "final_holdout_rmse": 260.117,
        "final_holdout_mape_percent": 9.994,
        "warning": (
            "Holt Damped performed best in the seven-record "
            "walk-forward comparison but did not outperform "
            "Naive in the 144-record single-origin holdout. "
            "It is a deployment candidate, not a universally "
            "validated best model."
        ),
    }

    metadata_path = (
        MODEL_DIR
        / "care_load_model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nMODEL ARTIFACT SAVED")
    print("-" * 70)

    print(model_path)
    print(metadata_path)

    print("\nMODEL SUMMARY")
    print("-" * 70)

    print(f"Selected candidate: Holt Damped")
    print(f"Latest observed value: {values[-1]:.2f}")

    print(
        "\nReminder: model performance depends on "
        "forecast horizon and evaluation design."
    )

    print("\nPHASE 4.2 COMPLETED")


if __name__ == "__main__":
    main()