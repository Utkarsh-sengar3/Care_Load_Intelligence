from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "uac_observed_records.csv"
)

REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COLUMNS = {
    "primary": "Children in HHS Care",
    "discharge": "Children discharged from HHS Care",
    "transfer": "Children transferred out of CBP custody",
    "cbp_care_load": "Children in CBP custody",
    "intake": "Children apprehended and placed in CBP custody*",
}


def main():
    print("\n" + "=" * 70)
    print("PHASE 1.5 - FORECASTING DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load and validate data
    # --------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required_columns = [
        "Date",
        *TARGET_COLUMNS.values(),
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df["Date"].isna().any():
        raise ValueError(
            "Date column contains missing values."
        )

    if df["Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected. Resolve them before modeling."
        )

    for column in TARGET_COLUMNS.values():
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if df[column].isna().any():
            raise ValueError(
                f"Missing values found in {column}"
            )

        if (df[column] < 0).any():
            raise ValueError(
                f"Negative values found in {column}"
            )

    print("\n1. DATASET VALIDATION")
    print("-" * 70)
    print(f"Observations: {len(df)}")
    print(f"Start date: {df['Date'].min().date()}")
    print(f"End date: {df['Date'].max().date()}")
    print("Duplicate dates: 0")
    print("Missing target values: 0")
    print("Negative target values: 0")

    # --------------------------------------------------
    # 2. Save original observed-date dataset
    # --------------------------------------------------
    observed_path = (
        PROCESSED_DIR
        / "uac_observed_forecasting_data.csv"
    )

    df.to_csv(observed_path, index=False)

    print("\n2. OBSERVED-DATE DATASET")
    print("-" * 70)
    print(f"Rows: {len(df)}")
    print(f"Saved: {observed_path.relative_to(ROOT)}")

    # --------------------------------------------------
    # 3. Create calendar-week aggregation
    # --------------------------------------------------
    weekly_source = df.copy()

    weekly_source["Week_Start"] = (
        weekly_source["Date"]
        - pd.to_timedelta(
            weekly_source["Date"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    # Count distinct observed dates in each week.
    weekly = (
        weekly_source.groupby("Week_Start")
        .agg(
            Observed_Days=("Date", "nunique"),
            First_Observed_Date=("Date", "min"),
            Last_Observed_Date=("Date", "max"),
            **{
                f"{column}_Mean": (column, "mean")
                for column in TARGET_COLUMNS.values()
            },
            **{
                f"{column}_Median": (column, "median")
                for column in TARGET_COLUMNS.values()
            },
        )
        .reset_index()
    )

    # Calculate the number of calendar days inside
    # the original dataset's date range for each week.
    first_date = df["Date"].min().normalize()
    last_date = df["Date"].max().normalize()

    week_end = (
        weekly["Week_Start"]
        + pd.Timedelta(days=6)
    )

    effective_start = weekly["Week_Start"].where(
        weekly["Week_Start"] >= first_date,
        first_date,
    )

    effective_end = week_end.where(
        week_end <= last_date,
        last_date,
    )

    weekly["Calendar_Days_In_Range"] = (
        effective_end - effective_start
    ).dt.days + 1

    weekly["Coverage_Percent"] = (
        weekly["Observed_Days"]
        / weekly["Calendar_Days_In_Range"]
        * 100
    ).round(2)

    weekly["Low_Coverage_Flag"] = (
        weekly["Coverage_Percent"] < 50
    )

    weekly["Full_Week_Coverage"] = (
        weekly["Coverage_Percent"] == 100
    )

    weekly_path = (
        PROCESSED_DIR
        / "uac_weekly_forecasting_data.csv"
    )

    weekly.to_csv(weekly_path, index=False)

    print("\n3. WEEKLY DATASET")
    print("-" * 70)
    print(f"Calendar weeks represented: {len(weekly)}")
    print(
        "Weeks with coverage below 50%:",
        int(weekly["Low_Coverage_Flag"].sum()),
    )
    print(
        "Weeks with complete coverage:",
        int(weekly["Full_Week_Coverage"].sum()),
    )
    print(
        "Mean weekly coverage:",
        round(weekly["Coverage_Percent"].mean(), 2),
        "%",
    )
    print(f"Saved: {weekly_path.relative_to(ROOT)}")

    # --------------------------------------------------
    # 4. Target-variable summary
    # --------------------------------------------------
    summary_rows = []

    for role, column in TARGET_COLUMNS.items():
        summary_rows.append({
            "Role": role,
            "Target": column,
            "Observed_Records": int(df[column].count()),
            "Mean": round(df[column].mean(), 2),
            "Median": round(df[column].median(), 2),
            "Minimum": df[column].min(),
            "Maximum": df[column].max(),
            "Standard_Deviation": round(
                df[column].std(),
                2,
            ),
        })

    target_summary = pd.DataFrame(summary_rows)

    target_summary_path = (
        REPORTS_DIR
        / "phase1_5_target_summary.csv"
    )

    target_summary.to_csv(
        target_summary_path,
        index=False,
    )

    print("\n4. TARGET SUMMARY")
    print("-" * 70)
    print(target_summary.to_string(index=False))

    # --------------------------------------------------
    # 5. Save coverage report
    # --------------------------------------------------
    coverage_report = weekly[
        [
            "Week_Start",
            "First_Observed_Date",
            "Last_Observed_Date",
            "Observed_Days",
            "Calendar_Days_In_Range",
            "Coverage_Percent",
            "Low_Coverage_Flag",
            "Full_Week_Coverage",
        ]
    ].copy()

    coverage_path = (
        REPORTS_DIR
        / "phase1_5_weekly_coverage.csv"
    )

    coverage_report.to_csv(
        coverage_path,
        index=False,
    )

    # --------------------------------------------------
    # 6. Final summary
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1.5 COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")
    print("data/processed/uac_observed_forecasting_data.csv")
    print("data/processed/uac_weekly_forecasting_data.csv")
    print("reports/phase1_5_target_summary.csv")
    print("reports/phase1_5_weekly_coverage.csv")

    print("\nOriginal observations were preserved.")
    print("Missing dates were not filled with zero.")
    print("No outliers were removed.")
    print(
        "Weekly means describe observed records only; "
        "they are not guaranteed seven-day averages."
    )


if __name__ == "__main__":
    main()