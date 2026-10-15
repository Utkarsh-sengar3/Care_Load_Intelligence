from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. PROJECT PATHS
# =========================================================

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


# =========================================================
# 2. CONFIGURATION
# =========================================================

TARGET_COLUMN = "Children in HHS Care"

# These are illustrative thresholds, NOT verified
# official HHS capacity limits.
CAPACITY_THRESHOLD = 5000

WARNING_UTILIZATION = 0.80
HIGH_RISK_UTILIZATION = 1.00


# =========================================================
# 3. LOAD AND VALIDATE DATA
# =========================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = ["Date", TARGET_COLUMN]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

df = df.dropna(
    subset=["Date", TARGET_COLUMN]
)

df = df.sort_values("Date").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid observations found.")

if (df[TARGET_COLUMN] < 0).any():
    raise ValueError(
        "Negative care-load values found. "
        "Check the dataset before continuing."
    )

if CAPACITY_THRESHOLD <= 0:
    raise ValueError(
        "Capacity threshold must be greater than zero."
    )


# =========================================================
# 4. CAPACITY PLANNING FUNCTIONS
# =========================================================

def classify_capacity_risk(
    care_load,
    capacity,
    warning_utilization=0.80,
    high_risk_utilization=1.00
):
    """
    Classify care-load risk using configurable thresholds.

    Low:
        Utilization below warning threshold.

    Medium:
        Utilization reaches warning threshold
        but remains below high-risk threshold.

    High:
        Utilization reaches or exceeds high-risk threshold.
    """

    utilization = care_load / capacity

    if utilization >= high_risk_utilization:
        return "High"

    if utilization >= warning_utilization:
        return "Medium"

    return "Low"


def estimate_additional_capacity(
    projected_load,
    capacity
):
    """
    Calculate additional places needed to accommodate
    projected demand above the configured capacity.
    """

    return max(
        0,
        int(np.ceil(projected_load - capacity))
    )


# =========================================================
# 5. HISTORICAL CAPACITY ANALYSIS
# =========================================================

df["Capacity_Threshold"] = CAPACITY_THRESHOLD

df["Capacity_Utilization"] = (
    df[TARGET_COLUMN] / CAPACITY_THRESHOLD
)

df["Capacity_Utilization_Percent"] = (
    df["Capacity_Utilization"] * 100
)

df["Additional_Capacity_Required"] = df[
    TARGET_COLUMN
].apply(
    lambda value: estimate_additional_capacity(
        value,
        CAPACITY_THRESHOLD
    )
)

df["Capacity_Risk"] = df[TARGET_COLUMN].apply(
    lambda value: classify_capacity_risk(
        care_load=value,
        capacity=CAPACITY_THRESHOLD,
        warning_utilization=WARNING_UTILIZATION,
        high_risk_utilization=HIGH_RISK_UTILIZATION
    )
)


# =========================================================
# 6. HISTORICAL SUMMARY
# =========================================================

care_load = df[TARGET_COLUMN]

summary = {
    "analysis_type": "Illustrative capacity scenario",
    "target": TARGET_COLUMN,
    "capacity_threshold": CAPACITY_THRESHOLD,
    "warning_utilization_percent": (
        WARNING_UTILIZATION * 100
    ),
    "high_risk_utilization_percent": (
        HIGH_RISK_UTILIZATION * 100
    ),
    "first_observation": str(
        df["Date"].min().date()
    ),
    "last_observation": str(
        df["Date"].max().date()
    ),
    "observations": int(len(df)),
    "mean_care_load": round(
        float(care_load.mean()), 2
    ),
    "median_care_load": round(
        float(care_load.median()), 2
    ),
    "minimum_care_load": int(
        care_load.min()
    ),
    "maximum_care_load": int(
        care_load.max()
    ),
    "mean_capacity_utilization_percent": round(
        float(
            df["Capacity_Utilization_Percent"].mean()
        ),
        2
    ),
    "maximum_capacity_utilization_percent": round(
        float(
            df["Capacity_Utilization_Percent"].max()
        ),
        2
    ),
    "observations_above_capacity": int(
        (care_load >= CAPACITY_THRESHOLD).sum()
    ),
    "observations_in_medium_risk": int(
        (df["Capacity_Risk"] == "Medium").sum()
    ),
    "observations_in_high_risk": int(
        (df["Capacity_Risk"] == "High").sum()
    ),
    "capacity_threshold_is_verified": False,
    "warning": (
        "The capacity threshold is illustrative. "
        "Results do not establish actual HHS capacity "
        "or actual operational risk."
    )
}


# =========================================================
# 7. RISK DISTRIBUTION
# =========================================================

risk_distribution = (
    df["Capacity_Risk"]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"],
        fill_value=0
    )
    .rename_axis("Capacity_Risk")
    .reset_index(name="Observation_Count")
)

risk_distribution["Percentage"] = (
    risk_distribution["Observation_Count"]
    / len(df)
    * 100
).round(2)


# =========================================================
# 8. MONTHLY SUMMARY
# =========================================================

df["Month"] = df["Date"].dt.to_period("M").astype(str)

monthly_summary = (
    df.groupby("Month")
    .agg(
        Observations=(TARGET_COLUMN, "count"),
        Mean_Care_Load=(TARGET_COLUMN, "mean"),
        Maximum_Care_Load=(TARGET_COLUMN, "max"),
        Minimum_Care_Load=(TARGET_COLUMN, "min"),
        Mean_Capacity_Utilization=(
            "Capacity_Utilization_Percent",
            "mean"
        ),
        High_Risk_Observations=(
            "Capacity_Risk",
            lambda values: (values == "High").sum()
        )
    )
    .reset_index()
)

monthly_summary[
    [
        "Mean_Care_Load",
        "Mean_Capacity_Utilization"
    ]
] = monthly_summary[
    [
        "Mean_Care_Load",
        "Mean_Capacity_Utilization"
    ]
].round(2)


# =========================================================
# 9. SAVE REPORTS
# =========================================================

historical_output = (
    REPORT_DIR
    / "phase5_1_historical_capacity_analysis.csv"
)

risk_output = (
    REPORT_DIR
    / "phase5_1_capacity_risk_distribution.csv"
)

monthly_output = (
    REPORT_DIR
    / "phase5_1_monthly_capacity_summary.csv"
)

summary_output = (
    REPORT_DIR
    / "phase5_1_capacity_summary.json"
)

df.to_csv(
    historical_output,
    index=False
)

risk_distribution.to_csv(
    risk_output,
    index=False
)

monthly_summary.to_csv(
    monthly_output,
    index=False
)

with open(
    summary_output,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        summary,
        file,
        indent=4
    )


# =========================================================
# 10. VISUALIZATION: CARE LOAD VS CAPACITY
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df[TARGET_COLUMN],
    label="Observed HHS Care Load"
)

plt.axhline(
    CAPACITY_THRESHOLD,
    linestyle="--",
    label="Illustrative Capacity Threshold"
)

plt.axhline(
    CAPACITY_THRESHOLD * WARNING_UTILIZATION,
    linestyle=":",
    label="Illustrative Warning Threshold"
)

plt.title(
    "Historical Care Load vs Illustrative Capacity"
)

plt.xlabel("Observation Date")
plt.ylabel("Children in HHS Care")

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "phase5_1_care_load_vs_capacity.png",
    dpi=200
)

plt.close()


# =========================================================
# 11. VISUALIZATION: RISK DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    risk_distribution["Capacity_Risk"],
    risk_distribution["Observation_Count"]
)

plt.title(
    "Historical Capacity Risk Distribution"
)

plt.xlabel("Illustrative Risk Category")
plt.ylabel("Number of Observations")

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "phase5_1_capacity_risk_distribution.png",
    dpi=200
)

plt.close()


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 60)
print("PHASE 5.1 — CAPACITY PLANNING ANALYSIS")
print("=" * 60)

print("\nIMPORTANT:")
print(
    "Capacity threshold is illustrative, "
    "not a verified official capacity."
)

print("\nHistorical Summary:")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nRisk Distribution:")
print(
    risk_distribution.to_string(index=False)
)

print("\nMonthly Summary — Latest 6 Months:")
print(
    monthly_summary.tail(6).to_string(index=False)
)

print("\nFiles Created:")
print(historical_output)
print(risk_output)
print(monthly_output)
print(summary_output)

print(
    FIGURE_DIR
    / "phase5_1_care_load_vs_capacity.png"
)

print(
    FIGURE_DIR
    / "phase5_1_capacity_risk_distribution.png"
)

print("\nPHASE 5.1 COMPLETE")