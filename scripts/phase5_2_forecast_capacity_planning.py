from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "uac_observed_forecasting_data.csv"
)

REPORT_DIR = ROOT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Children in HHS Care"

# Illustrative scenario only; not verified official capacity.
CAPACITY = 5000

WARNING_UTILIZATION = 0.80

# Forecast 7 future observed records.
FORECAST_HORIZON = 7


# =====================================================
# 2. LOAD DATA
# =====================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = ["Date", TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

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

if len(df) < 50:
    raise ValueError(
        "Insufficient observations for model fitting."
    )

if (df[TARGET] < 0).any():
    raise ValueError(
        "Negative target values detected."
    )

if CAPACITY <= 0:
    raise ValueError(
        "Capacity must be greater than zero."
    )

series = df[TARGET].astype(float)


# =====================================================
# 3. FIT FORECAST MODELS
# =====================================================

print("\nFitting forecasting models...")

models = {}

# Model 1: Naive
models["Naive"] = np.repeat(
    series.iloc[-1],
    FORECAST_HORIZON
)

# Model 2: ARIMA
try:
    arima_model = ARIMA(
        series,
        order=(1, 1, 1)
    ).fit()

    models["ARIMA_1_1_1"] = np.asarray(
        arima_model.forecast(
            steps=FORECAST_HORIZON
        )
    )

except Exception as error:
    print(f"ARIMA could not be fitted: {error}")


# Model 3: Holt Damped Trend
try:
    holt_model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    models["Holt_Damped"] = np.asarray(
        holt_model.forecast(
            FORECAST_HORIZON
        )
    )

except Exception as error:
    print(f"Holt Damped could not be fitted: {error}")


if not models:
    raise RuntimeError(
        "No forecasting model could be fitted."
    )


# =====================================================
# 4. SELECT MODEL FOR THIS SCENARIO
# =====================================================

# The long single-origin holdout showed Naive and ARIMA
# performed almost identically and better than Holt Damped.
#
# The short walk-forward evaluation favored Holt Damped.
#
# To avoid declaring one universal winner, expose model
# selection as a configurable setting.

SELECTED_MODEL = "Naive"

if SELECTED_MODEL not in models:
    raise ValueError(
        f"Selected model {SELECTED_MODEL} is unavailable. "
        f"Available models: {list(models.keys())}"
    )

forecast_values = np.asarray(
    models[SELECTED_MODEL],
    dtype=float
)

# Care-load forecasts cannot be negative.
forecast_values = np.maximum(
    forecast_values,
    0
)


# =====================================================
# 5. CREATE FORECAST TABLE
# =====================================================

last_observed_date = df["Date"].max()

forecast_df = pd.DataFrame({
    "Forecast_Step": np.arange(
        1,
        FORECAST_HORIZON + 1
    ),
    "Forecast_Care_Load": np.round(
        forecast_values
    ).astype(int)
})

# These are step numbers, not dates.
# We do not invent calendar dates for irregular records.

forecast_df["Configured_Capacity"] = CAPACITY

forecast_df["Capacity_Utilization_Percent"] = (
    forecast_df["Forecast_Care_Load"]
    / CAPACITY
    * 100
).round(2)

forecast_df["Warning_Threshold"] = (
    CAPACITY * WARNING_UTILIZATION
)

forecast_df["Estimated_Additional_Places"] = (
    forecast_df["Forecast_Care_Load"] - CAPACITY
).clip(lower=0).apply(np.ceil).astype(int)


def classify_risk(load, capacity):
    utilization = load / capacity

    if utilization >= 1:
        return "High"

    if utilization >= WARNING_UTILIZATION:
        return "Medium"

    return "Low"


forecast_df["Scenario_Risk"] = forecast_df[
    "Forecast_Care_Load"
].apply(
    lambda value: classify_risk(
        value,
        CAPACITY
    )
)

forecast_df["Last_Observed_Date"] = (
    last_observed_date.strftime("%Y-%m-%d")
)

forecast_df["Selected_Model"] = SELECTED_MODEL


# =====================================================
# 6. WHAT-IF CAPACITY SCENARIOS
# =====================================================

scenario_capacities = [
    4000,
    5000,
    6000,
    8000,
    10000
]

scenario_rows = []

for capacity in scenario_capacities:
    for _, row in forecast_df.iterrows():

        projected_load = int(
            row["Forecast_Care_Load"]
        )

        utilization = (
            projected_load / capacity * 100
        )

        if utilization >= 100:
            risk = "High"
        elif utilization >= WARNING_UTILIZATION * 100:
            risk = "Medium"
        else:
            risk = "Low"

        scenario_rows.append({
            "Forecast_Step": int(
                row["Forecast_Step"]
            ),
            "Forecast_Care_Load": projected_load,
            "Scenario_Capacity": capacity,
            "Utilization_Percent": round(
                utilization,
                2
            ),
            "Estimated_Additional_Places": max(
                0,
                int(
                    np.ceil(
                        projected_load - capacity
                    )
                )
            ),
            "Scenario_Risk": risk
        })

scenario_df = pd.DataFrame(
    scenario_rows
)


# =====================================================
# 7. SUMMARY
# =====================================================

summary = {
    "analysis": "Forecast-based capacity scenario",
    "last_observed_date": str(
        last_observed_date.date()
    ),
    "historical_observations": int(len(df)),
    "selected_model": SELECTED_MODEL,
    "forecast_horizon_observed_records": (
        FORECAST_HORIZON
    ),
    "configured_capacity": CAPACITY,
    "warning_utilization_percent": (
        WARNING_UTILIZATION * 100
    ),
    "forecast_minimum": int(
        forecast_df["Forecast_Care_Load"].min()
    ),
    "forecast_maximum": int(
        forecast_df["Forecast_Care_Load"].max()
    ),
    "forecast_mean": round(
        float(
            forecast_df["Forecast_Care_Load"].mean()
        ),
        2
    ),
    "maximum_estimated_additional_places": int(
        forecast_df[
            "Estimated_Additional_Places"
        ].max()
    ),
    "capacity_verified": False,
    "forecast_uncertainty_interval_included": False,
    "warning": (
        "These are point forecasts for future observed records, "
        "not guaranteed calendar-day forecasts. Capacity is "
        "illustrative, and the table does not include calibrated "
        "forecast uncertainty intervals."
    )
}


# =====================================================
# 8. SAVE OUTPUTS
# =====================================================

forecast_path = (
    REPORT_DIR
    / "phase5_2_forecast_capacity_plan.csv"
)

scenario_path = (
    REPORT_DIR
    / "phase5_2_capacity_scenarios.csv"
)

summary_path = (
    REPORT_DIR
    / "phase5_2_forecast_capacity_summary.json"
)

forecast_df.to_csv(
    forecast_path,
    index=False
)

scenario_df.to_csv(
    scenario_path,
    index=False
)

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
# 9. DISPLAY RESULTS
# =====================================================

print("\n" + "=" * 65)
print("PHASE 5.2 — FORECAST-BASED CAPACITY PLANNING")
print("=" * 65)

print(f"\nLast observed date: {last_observed_date.date()}")
print(f"Selected model: {SELECTED_MODEL}")
print(f"Configured capacity: {CAPACITY}")
print(
    f"Forecast horizon: {FORECAST_HORIZON} "
    "future observed records"
)

print("\nForecast Capacity Plan:")
print(
    forecast_df[
        [
            "Forecast_Step",
            "Forecast_Care_Load",
            "Capacity_Utilization_Percent",
            "Estimated_Additional_Places",
            "Scenario_Risk"
        ]
    ].to_string(index=False)
)

print("\nCapacity Scenario Comparison:")
print(
    scenario_df.to_string(index=False)
)

print("\nSummary:")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFiles Created:")
print(forecast_path)
print(scenario_path)
print(summary_path)

print("\nPHASE 5.2 COMPLETE")