from pathlib import Path
import json

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# =====================================================
# PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "uac_observed_forecasting_data.csv"
)

DEFAULT_PREDICTIONS_FILE = (
    ROOT_DIR
    / "reports"
    / "phase3_4_advanced_predictions.csv"
)

TARGET_COLUMN = "Children in HHS Care"


# =====================================================
# DATA LOADING
# =====================================================

def load_care_data(data_path=DEFAULT_DATA_FILE):
    """Load and validate the historical care-load dataset."""

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    required = ["Date", TARGET_COLUMN]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    df = (
        df.dropna(subset=["Date", TARGET_COLUMN])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"])
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise ValueError(
            "At least 100 observations are required."
        )

    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError(
            "Negative care-load values detected."
        )

    return df


# =====================================================
# FORECASTING
# =====================================================

def generate_forecast(
    df,
    model_name="Holt_Damped",
    horizon=7
):
    """
    Generate future observed-record forecasts.

    Supported models:
        Naive
        ARIMA_1_1_1
        Holt_Damped

    Horizon represents future observed records,
    not calendar days.
    """

    supported_models = [
        "Naive",
        "ARIMA_1_1_1",
        "Holt_Damped"
    ]

    if model_name not in supported_models:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Choose from {supported_models}."
        )

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(
            "Horizon must be a positive integer."
        )

    series = df[TARGET_COLUMN].astype(float)

    if model_name == "Naive":

        predictions = np.repeat(
            series.iloc[-1],
            horizon
        )

    elif model_name == "ARIMA_1_1_1":

        fitted_model = ARIMA(
            series,
            order=(1, 1, 1)
        ).fit()

        predictions = np.asarray(
            fitted_model.forecast(
                steps=horizon
            )
        )

    else:

        fitted_model = ExponentialSmoothing(
            series,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True)

        predictions = np.asarray(
            fitted_model.forecast(
                steps=horizon
            )
        )

    predictions = np.maximum(
        predictions,
        0
    )

    return pd.DataFrame({
        "Forecast_Step": np.arange(
            1,
            horizon + 1
        ),
        "Forecast_Care_Load": predictions
    })


# =====================================================
# EMPIRICAL UNCERTAINTY
# =====================================================

def add_empirical_error_ranges(
    forecast_df,
    model_name,
    predictions_path=DEFAULT_PREDICTIONS_FILE
):
    """
    Add empirical absolute-error ranges based on the
    existing walk-forward predictions.

    These are not guaranteed prediction intervals.
    """

    predictions_path = Path(predictions_path)

    result = forecast_df.copy()

    result["Lower_Q80"] = np.nan
    result["Upper_Q80"] = np.nan

    result["Lower_Q90"] = np.nan
    result["Upper_Q90"] = np.nan

    result["Lower_Q95"] = np.nan
    result["Upper_Q95"] = np.nan

    if not predictions_path.exists():
        result["Uncertainty_Available"] = False

        return result

    history = pd.read_csv(
        predictions_path
    )

    required = [
        "Model",
        "Actual",
        "Prediction",
        "Forecast_Step"
    ]

    if any(
        column not in history.columns
        for column in required
    ):
        result["Uncertainty_Available"] = False

        return result

    # The Phase 3.4 evaluation uses these exact model labels.
    history = history[
        history["Model"] == model_name
    ].copy()

    if history.empty:
        result["Uncertainty_Available"] = False

        return result

    history["Absolute_Error"] = (
        history["Actual"] - history["Prediction"]
    ).abs()

    for index, row in result.iterrows():

        step = int(
            row["Forecast_Step"]
        )

        errors = history.loc[
            history["Forecast_Step"] == step,
            "Absolute_Error"
        ].dropna()

        if len(errors) < 10:
            continue

        q80 = float(
            errors.quantile(0.80)
        )

        q90 = float(
            errors.quantile(0.90)
        )

        q95 = float(
            errors.quantile(0.95)
        )

        prediction = float(
            row["Forecast_Care_Load"]
        )

        result.loc[index, "Lower_Q80"] = max(
            0,
            prediction - q80
        )

        result.loc[index, "Upper_Q80"] = (
            prediction + q80
        )

        result.loc[index, "Lower_Q90"] = max(
            0,
            prediction - q90
        )

        result.loc[index, "Upper_Q90"] = (
            prediction + q90
        )

        result.loc[index, "Lower_Q95"] = max(
            0,
            prediction - q95
        )

        result.loc[index, "Upper_Q95"] = (
            prediction + q95
        )

    result["Uncertainty_Available"] = (
        result["Lower_Q90"].notna()
        & result["Upper_Q90"].notna()
    )

    return result


# =====================================================
# CAPACITY PLANNING
# =====================================================

def add_capacity_analysis(
    forecast_df,
    capacity=5000,
    warning_utilization=0.80
):
    """Add configurable capacity and stress scenarios."""

    if capacity <= 0:
        raise ValueError(
            "Capacity must be greater than zero."
        )

    if not 0 < warning_utilization < 1:
        raise ValueError(
            "Warning utilization must be between 0 and 1."
        )

    result = forecast_df.copy()

    result["Configured_Capacity"] = capacity

    result["Capacity_Utilization_Percent"] = (
        result["Forecast_Care_Load"]
        / capacity
        * 100
    ).round(2)

    result["Estimated_Additional_Places"] = (
        result["Forecast_Care_Load"] - capacity
    ).clip(lower=0).apply(np.ceil).astype(int)

    warning_threshold = (
        capacity * warning_utilization
    )

    def classify_risk(value):

        utilization = value / capacity

        if utilization >= 1:
            return "High"

        if utilization >= warning_utilization:
            return "Medium"

        return "Low"

    result["Scenario_Risk"] = (
        result["Forecast_Care_Load"]
        .apply(classify_risk)
    )

    result["Warning_Threshold"] = (
        warning_threshold
    )

    return result


# =====================================================
# DEMAND STRESS SCENARIOS
# =====================================================

def generate_stress_scenarios(
    forecast_df,
    capacity=5000,
    warning_utilization=0.80
):
    """Generate hypothetical demand-increase scenarios."""

    scenarios = {
        "Baseline": 1.00,
        "Elevated_Demand": 1.10,
        "High_Demand_Stress": 1.25
    }

    rows = []

    for _, forecast in forecast_df.iterrows():

        base_load = float(
            forecast["Forecast_Care_Load"]
        )

        for scenario_name, multiplier in (
            scenarios.items()
        ):

            scenario_load = int(
                np.ceil(
                    base_load * multiplier
                )
            )

            utilization = (
                scenario_load / capacity
            )

            if utilization >= 1:
                risk = "High"

            elif utilization >= warning_utilization:
                risk = "Medium"

            else:
                risk = "Low"

            rows.append({
                "Forecast_Step": int(
                    forecast["Forecast_Step"]
                ),
                "Scenario": scenario_name,
                "Multiplier": multiplier,
                "Projected_Care_Load": scenario_load,
                "Capacity": capacity,
                "Utilization_Percent": round(
                    utilization * 100,
                    2
                ),
                "Estimated_Additional_Places": max(
                    0,
                    scenario_load - capacity
                ),
                "Scenario_Risk": risk
            })

    return pd.DataFrame(rows)


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_forecasting_pipeline(
    model_name="Holt_Damped",
    horizon=7,
    capacity=5000,
    warning_utilization=0.80
):
    """
    Unified entry point for the Streamlit application.
    """

    df = load_care_data()

    forecasts = generate_forecast(
        df=df,
        model_name=model_name,
        horizon=horizon
    )

    forecasts = add_empirical_error_ranges(
        forecast_df=forecasts,
        model_name=model_name
    )

    forecasts = add_capacity_analysis(
        forecast_df=forecasts,
        capacity=capacity,
        warning_utilization=warning_utilization
    )

    scenarios = generate_stress_scenarios(
        forecast_df=forecasts,
        capacity=capacity,
        warning_utilization=warning_utilization
    )

    summary = {
        "model": model_name,
        "last_observed_date": str(
            df["Date"].max().date()
        ),
        "historical_observations": int(
            len(df)
        ),
        "forecast_horizon_observed_records": (
            horizon
        ),
        "latest_observed_care_load": int(
            df[TARGET_COLUMN].iloc[-1]
        ),
        "capacity_threshold": capacity,
        "capacity_verified": False,
        "mean_forecast": round(
            float(
                forecasts["Forecast_Care_Load"].mean()
            ),
            2
        ),
        "maximum_forecast": round(
            float(
                forecasts["Forecast_Care_Load"].max()
            ),
            2
        ),
        "maximum_estimated_additional_places": int(
            forecasts[
                "Estimated_Additional_Places"
            ].max()
        ),
        "uncertainty_available": bool(
            forecasts["Uncertainty_Available"].any()
        ),
        "uncertainty_is_guaranteed": False,
        "forecast_horizon_unit": (
            "Future observed records, not calendar days"
        )
    }

    return forecasts, scenarios, summary


# =====================================================
# LOCAL TEST
# =====================================================

if __name__ == "__main__":

    forecasts, scenarios, summary = (
        run_forecasting_pipeline(
            model_name="Holt_Damped",
            horizon=7,
            capacity=5000,
            warning_utilization=0.80
        )
    )

    report_dir = ROOT_DIR / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    forecasts.to_csv(
        report_dir / "phase5_4_unified_forecast.csv",
        index=False
    )

    scenarios.to_csv(
        report_dir / "phase5_4_unified_scenarios.csv",
        index=False
    )

    with open(
        report_dir / "phase5_4_unified_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("PHASE 5.4 — UNIFIED FORECASTING ENGINE")
    print("=" * 60)

    print("\nForecast:")
    print(
        forecasts.to_string(index=False)
    )

    print("\nStress Scenarios:")
    print(
        scenarios.to_string(index=False)
    )

    print("\nSummary:")
    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    print("\nPHASE 5.4 COMPLETE")