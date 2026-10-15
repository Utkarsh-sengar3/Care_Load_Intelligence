from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "uac_observed_forecasting_data.csv"
)

REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"

FORECAST_HORIZON = 7

# Illustrative only: not verified official capacity.
CAPACITY = 5000

WARNING_UTILIZATION = 0.80


# =====================================================
# 2. LOAD DATA
# =====================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

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

if len(df) < 100:
    raise ValueError(
        "At least 100 valid observations are required."
    )

series = df[TARGET].astype(float)

if (series < 0).any():
    raise ValueError(
        "Negative care-load observations detected."
    )

last_observed_date = df["Date"].max()


# =====================================================
# 3. WALK-FORWARD ERROR DATA
# =====================================================

# Reuse the previously evaluated predictions.
# This avoids treating training residuals as genuine
# out-of-sample forecasting errors.

PREDICTIONS_FILE = (
    REPORT_DIR
    / "phase3_4_advanced_predictions.csv"
)

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        "Required file not found:\n"
        f"{PREDICTIONS_FILE}\n"
        "Run Phase 3.4 first."
    )

historical_predictions = pd.read_csv(
    PREDICTIONS_FILE
)

required_prediction_columns = [
    "Model",
    "Actual",
    "Prediction",
    "Forecast_Step"
]

missing_columns = [
    column
    for column in required_prediction_columns
    if column not in historical_predictions.columns
]

if missing_columns:
    raise ValueError(
        "The Phase 3.4 predictions file is missing "
        f"these columns: {missing_columns}"
    )

historical_predictions["Error"] = (
    historical_predictions["Actual"]
    - historical_predictions["Prediction"]
)

historical_predictions["Absolute_Error"] = (
    historical_predictions["Error"].abs()
)


# =====================================================
# 4. FIT FORECAST MODELS
# =====================================================

forecasts = {}

# Model 1: Naive
forecasts["Naive"] = np.repeat(
    series.iloc[-1],
    FORECAST_HORIZON
)

# Model 2: ARIMA
try:
    arima_model = ARIMA(
        series,
        order=(1, 1, 1)
    ).fit()

    forecasts["ARIMA_1_1_1"] = np.asarray(
        arima_model.forecast(
            steps=FORECAST_HORIZON
        )
    )

except Exception as error:
    print("ARIMA fitting warning:", error)

# Model 3: Holt Damped
try:
    holt_model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    forecasts["Holt_Damped"] = np.asarray(
        holt_model.forecast(
            steps=FORECAST_HORIZON
        )
    )

except Exception as error:
    print("Holt Damped fitting warning:", error)

if not forecasts:
    raise RuntimeError(
        "No model could generate forecasts."
    )


# =====================================================
# 5. FORECAST TABLE
# =====================================================

forecast_rows = []

for model_name, values in forecasts.items():

    values = np.maximum(
        np.asarray(values, dtype=float),
        0
    )

    for step, prediction in enumerate(values, start=1):

        forecast_rows.append({
            "Model": model_name,
            "Forecast_Step": step,
            "Forecast_Care_Load": float(prediction),
            "Configured_Capacity": CAPACITY,
            "Last_Observed_Date": (
                last_observed_date.strftime("%Y-%m-%d")
            )
        })

forecast_df = pd.DataFrame(forecast_rows)


# =====================================================
# 6. HISTORICAL OUT-OF-SAMPLE ERROR QUANTILES
# =====================================================

# Use absolute errors from the earlier walk-forward
# evaluation. These are empirical error ranges, not
# guaranteed statistical prediction intervals.

interval_rows = []

for model_name, values in forecasts.items():

    model_errors = historical_predictions[
        historical_predictions["Model"] == model_name
    ].copy()

    if model_errors.empty:
        print(
            f"Warning: no historical errors for {model_name}. "
            "Uncertainty ranges will not be generated."
        )
        continue

    for step, prediction in enumerate(values, start=1):

        step_errors = model_errors.loc[
            model_errors["Forecast_Step"] == step,
            "Absolute_Error"
        ].dropna()

        if len(step_errors) < 10:
            continue

        # Empirical absolute-error quantiles.
        q80 = float(step_errors.quantile(0.80))
        q90 = float(step_errors.quantile(0.90))
        q95 = float(step_errors.quantile(0.95))

        interval_rows.append({
            "Model": model_name,
            "Forecast_Step": step,
            "Forecast_Care_Load": float(
                max(0, prediction)
            ),
            "Historical_Abs_Error_Q80": q80,
            "Historical_Abs_Error_Q90": q90,
            "Historical_Abs_Error_Q95": q95,
            "Lower_Q80": max(0, prediction - q80),
            "Upper_Q80": prediction + q80,
            "Lower_Q90": max(0, prediction - q90),
            "Upper_Q90": prediction + q90,
            "Lower_Q95": max(0, prediction - q95),
            "Upper_Q95": prediction + q95
        })

interval_df = pd.DataFrame(interval_rows)


# =====================================================
# 7. CAPACITY STRESS TESTING
# =====================================================

scenario_rows = []

# Scenario multipliers are illustrative stress tests.
scenario_multipliers = {
    "Baseline": 1.00,
    "Elevated_Demand": 1.10,
    "High_Demand_Stress": 1.25
}

for _, row in forecast_df.iterrows():

    base_prediction = row["Forecast_Care_Load"]

    for scenario_name, multiplier in (
        scenario_multipliers.items()
    ):

        scenario_load = int(
            np.ceil(base_prediction * multiplier)
        )

        utilization = (
            scenario_load / CAPACITY
        ) * 100

        additional_places = max(
            0,
            scenario_load - CAPACITY
        )

        if utilization >= 100:
            risk = "High"
        elif utilization >= WARNING_UTILIZATION * 100:
            risk = "Medium"
        else:
            risk = "Low"

        scenario_rows.append({
            "Model": row["Model"],
            "Forecast_Step": int(
                row["Forecast_Step"]
            ),
            "Scenario": scenario_name,
            "Demand_Multiplier": multiplier,
            "Scenario_Care_Load": scenario_load,
            "Capacity": CAPACITY,
            "Utilization_Percent": round(
                utilization,
                2
            ),
            "Estimated_Additional_Places": (
                additional_places
            ),
            "Scenario_Risk": risk
        })

scenario_df = pd.DataFrame(
    scenario_rows
)


# =====================================================
# 8. MODEL COMPARISON SUMMARY
# =====================================================

comparison_rows = []

for model_name in forecasts:

    model_errors = historical_predictions[
        historical_predictions["Model"] == model_name
    ].copy()

    if model_errors.empty:
        continue

    actual = model_errors["Actual"].astype(float)
    prediction = model_errors["Prediction"].astype(float)

    mae = float(
        np.mean(np.abs(actual - prediction))
    )

    rmse = float(
        np.sqrt(np.mean((actual - prediction) ** 2))
    )

    nonzero = actual != 0

    mape = (
        float(
            np.mean(
                np.abs(
                    (
                        actual[nonzero]
                        - prediction[nonzero]
                    )
                    / actual[nonzero]
                )
            ) * 100
        )
        if nonzero.any()
        else None
    )

    comparison_rows.append({
        "Model": model_name,
        "Historical_Walk_Forward_MAE": round(mae, 3),
        "Historical_Walk_Forward_RMSE": round(rmse, 3),
        "Historical_Walk_Forward_MAPE_Percent": (
            round(mape, 3)
            if mape is not None
            else None
        ),
        "Historical_Forecast_Count": len(model_errors)
    })

comparison_df = pd.DataFrame(
    comparison_rows
)


# =====================================================
# 9. SAVE OUTPUTS
# =====================================================

forecast_path = (
    REPORT_DIR
    / "phase5_3_model_forecasts.csv"
)

interval_path = (
    REPORT_DIR
    / "phase5_3_empirical_uncertainty_ranges.csv"
)

scenario_path = (
    REPORT_DIR
    / "phase5_3_demand_stress_scenarios.csv"
)

comparison_path = (
    REPORT_DIR
    / "phase5_3_model_comparison.csv"
)

summary_path = (
    REPORT_DIR
    / "phase5_3_summary.json"
)

forecast_df.to_csv(
    forecast_path,
    index=False
)

interval_df.to_csv(
    interval_path,
    index=False
)

scenario_df.to_csv(
    scenario_path,
    index=False
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

summary = {
    "analysis": "Forecast uncertainty and demand stress testing",
    "last_observed_date": str(
        last_observed_date.date()
    ),
    "historical_observations": int(len(df)),
    "forecast_horizon_observed_records": (
        FORECAST_HORIZON
    ),
    "models_forecasted": list(forecasts.keys()),
    "capacity_threshold": CAPACITY,
    "capacity_verified": False,
    "stress_scenario_multipliers": (
        scenario_multipliers
    ),
    "uncertainty_method": (
        "Empirical quantiles of absolute walk-forward "
        "forecast errors by horizon"
    ),
    "intervals_are_guaranteed": False,
    "limitations": [
        "Observation dates are irregular.",
        "Forecast steps represent future observed records, not calendar days.",
        "Capacity threshold is illustrative, not verified official capacity.",
        "Stress scenario multipliers are hypothetical, not estimated probabilities.",
        "Historical error quantiles are not guaranteed prediction intervals."
    ]
}

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )


# =====================================================
# 10. VISUALIZATION 1: MODEL FORECAST COMPARISON
# =====================================================

plt.figure(figsize=(11, 6))

for model_name, values in forecasts.items():

    plt.plot(
        range(1, FORECAST_HORIZON + 1),
        np.maximum(values, 0),
        marker="o",
        label=model_name
    )

plt.axhline(
    CAPACITY,
    linestyle="--",
    label="Illustrative Capacity"
)

plt.axhline(
    CAPACITY * WARNING_UTILIZATION,
    linestyle=":",
    label="Illustrative Warning Threshold"
)

plt.title("Future Care-Load Forecast Comparison")
plt.xlabel("Forecast Step (Future Observed Record)")
plt.ylabel("Predicted Children in HHS Care")

plt.xticks(
    range(1, FORECAST_HORIZON + 1)
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "phase5_3_model_forecast_comparison.png",
    dpi=200
)

plt.close()


# =====================================================
# 11. VISUALIZATION 2: FORECAST UNCERTAINTY
# =====================================================

if not interval_df.empty:

    # Show Holt Damped if available, otherwise first model.
    selected_interval_model = (
        "Holt_Damped"
        if "Holt_Damped" in interval_df["Model"].values
        else interval_df["Model"].iloc[0]
    )

    plot_df = interval_df[
        interval_df["Model"] == selected_interval_model
    ].sort_values("Forecast_Step")

    plt.figure(figsize=(11, 6))

    x = plot_df["Forecast_Step"]

    plt.plot(
        x,
        plot_df["Forecast_Care_Load"],
        marker="o",
        label="Forecast"
    )

    plt.fill_between(
        x,
        plot_df["Lower_Q90"],
        plot_df["Upper_Q90"],
        alpha=0.20,
        label="Empirical Q90 Error Range"
    )

    plt.axhline(
        CAPACITY,
        linestyle="--",
        label="Illustrative Capacity"
    )

    plt.title(
        f"Forecast and Empirical Error Range: "
        f"{selected_interval_model}"
    )

    plt.xlabel(
        "Forecast Step (Future Observed Record)"
    )

    plt.ylabel("Children in HHS Care")

    plt.xticks(
        range(1, FORECAST_HORIZON + 1)
    )

    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "phase5_3_forecast_uncertainty.png",
        dpi=200
    )

    plt.close()


# =====================================================
# 12. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 65)
print("PHASE 5.3 — FORECAST UNCERTAINTY & STRESS TESTING")
print("=" * 65)

print("\nModel Comparison:")
print(comparison_df.to_string(index=False))

print("\nForecasts:")
print(
    forecast_df[
        [
            "Model",
            "Forecast_Step",
            "Forecast_Care_Load"
        ]
    ].to_string(index=False)
)

print("\nEmpirical Uncertainty Ranges:")
if interval_df.empty:
    print("No uncertainty ranges generated.")
else:
    print(
        interval_df[
            [
                "Model",
                "Forecast_Step",
                "Forecast_Care_Load",
                "Lower_Q90",
                "Upper_Q90"
            ]
        ].to_string(index=False)
    )

print("\nDemand Stress Scenarios:")
print(scenario_df.to_string(index=False))

print("\nFiles Created:")
print(forecast_path)
print(interval_path)
print(scenario_path)
print(comparison_path)
print(summary_path)

print(
    FIGURE_DIR
    / "phase5_3_model_forecast_comparison.png"
)

print(
    FIGURE_DIR
    / "phase5_3_forecast_uncertainty.png"
)

print("\nPHASE 5.3 COMPLETE")