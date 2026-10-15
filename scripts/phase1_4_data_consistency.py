from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "interim" / "uac_observed_records.csv"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "=" * 70)
    print("PHASE 1.4 - DATA CONSISTENCY & FORECASTING FEASIBILITY")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    value_columns = [
        column for column in df.columns
        if column != "Date"
    ]

    df = df.sort_values("Date").reset_index(drop=True)

    print("\n1. DATASET OVERVIEW")
    print("-" * 70)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date: {df['Date'].max().date()}")
    print(f"Duplicate dates: {df['Date'].duplicated().sum()}")
    print(f"Missing dates in Date column: {df['Date'].isna().sum()}")

    # --------------------------------------------------
    # 2. Data quality checks
    # --------------------------------------------------
    quality_rows = []

    for column in value_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")

        quality_rows.append({
            "Column": column,
            "Data_Type": str(df[column].dtype),
            "Total_Rows": len(df),
            "Missing_Values": int(numeric.isna().sum()),
            "Negative_Values": int((numeric < 0).sum()),
            "Zero_Values": int((numeric == 0).sum()),
            "Unique_Values": int(numeric.nunique()),
            "Minimum": numeric.min(),
            "Maximum": numeric.max(),
        })

    quality_df = pd.DataFrame(quality_rows)

    quality_df.to_csv(
        REPORTS_DIR / "phase1_4_data_quality.csv",
        index=False,
    )

    print("\n2. DATA QUALITY SUMMARY")
    print("-" * 70)
    print(quality_df.to_string(index=False))

    # --------------------------------------------------
    # 3. Descriptive statistics
    # --------------------------------------------------
    numeric_df = df[value_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    stats = numeric_df.describe().T

    stats["Median"] = numeric_df.median()
    stats["Missing_Values"] = numeric_df.isna().sum()
    stats["Zero_Count"] = numeric_df.eq(0).sum()

    stats = stats.reset_index().rename(
        columns={"index": "Variable"}
    )

    stats.to_csv(
        REPORTS_DIR / "phase1_4_descriptive_statistics.csv",
        index=False,
    )

    print("\n3. DESCRIPTIVE STATISTICS")
    print("-" * 70)
    print(stats.round(2).to_string(index=False))

    # --------------------------------------------------
    # 4. Unusual values using IQR
    # --------------------------------------------------
    outlier_rows = []

    for column in value_columns:
        series = numeric_df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            numeric_df[column].notna()
            & (
                (numeric_df[column] < lower_bound)
                | (numeric_df[column] > upper_bound)
            )
        )

        flagged = df.loc[mask, ["Date", column]].copy()

        for _, row in flagged.iterrows():
            outlier_rows.append({
                "Date": row["Date"],
                "Variable": column,
                "Value": row[column],
                "IQR_Lower_Bound": lower_bound,
                "IQR_Upper_Bound": upper_bound,
            })

    outliers_df = pd.DataFrame(
        outlier_rows,
        columns=[
            "Date",
            "Variable",
            "Value",
            "IQR_Lower_Bound",
            "IQR_Upper_Bound",
        ],
    )

    outliers_df.to_csv(
        REPORTS_DIR / "phase1_4_potential_outliers.csv",
        index=False,
    )

    print("\n4. POTENTIAL OUTLIERS")
    print("-" * 70)
    print(f"Total flagged observations: {len(outliers_df)}")

    if not outliers_df.empty:
        print(
            outliers_df.head(15).round(2).to_string(index=False)
        )
    else:
        print("No observations flagged by the IQR rule.")

    print(
        "\nNote: IQR flags are investigation candidates, "
        "not proof of incorrect data."
    )

    # --------------------------------------------------
    # 5. Changes between consecutive observed records
    # --------------------------------------------------
    changes = df[["Date"]].copy()
    changes["Gap_Days"] = df["Date"].diff().dt.days

    change_summary_rows = []

    for column in value_columns:
        values = numeric_df[column]

        changes[f"{column}_Change"] = values.diff()

        changes[f"{column}_Absolute_Change"] = (
            values.diff().abs()
        )

        change_series = values.diff().dropna()

        change_summary_rows.append({
            "Variable": column,
            "Mean_Absolute_Change": (
                change_series.abs().mean()
                if not change_series.empty else np.nan
            ),
            "Median_Absolute_Change": (
                change_series.abs().median()
                if not change_series.empty else np.nan
            ),
            "Maximum_Absolute_Change": (
                change_series.abs().max()
                if not change_series.empty else np.nan
            ),
        })

    changes.to_csv(
        REPORTS_DIR / "phase1_4_observation_changes.csv",
        index=False,
    )

    change_summary = pd.DataFrame(change_summary_rows)

    change_summary.to_csv(
        REPORTS_DIR / "phase1_4_change_summary.csv",
        index=False,
    )

    print("\n5. CHANGE SUMMARY")
    print("-" * 70)
    print(change_summary.round(2).to_string(index=False))

    # --------------------------------------------------
    # 6. Correlation analysis
    # --------------------------------------------------
    correlation = numeric_df.corr()

    correlation.to_csv(
        REPORTS_DIR / "phase1_4_correlation_matrix.csv"
    )

    print("\n6. CORRELATION MATRIX")
    print("-" * 70)
    print(correlation.round(3).to_string())

    # --------------------------------------------------
    # 7. Duplicate and missing-value details
    # --------------------------------------------------
    duplicate_rows = df[
        df["Date"].duplicated(keep=False)
    ].copy()

    duplicate_rows.to_csv(
        REPORTS_DIR / "phase1_4_duplicate_dates.csv",
        index=False,
    )

    missing_value_rows = df[
        df.isna().any(axis=1)
    ].copy()

    missing_value_rows.to_csv(
        REPORTS_DIR / "phase1_4_rows_with_missing_values.csv",
        index=False,
    )

    # --------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1.4 OUTPUTS")
    print("=" * 70)

    print("reports/phase1_4_data_quality.csv")
    print("reports/phase1_4_descriptive_statistics.csv")
    print("reports/phase1_4_potential_outliers.csv")
    print("reports/phase1_4_observation_changes.csv")
    print("reports/phase1_4_change_summary.csv")
    print("reports/phase1_4_correlation_matrix.csv")
    print("reports/phase1_4_duplicate_dates.csv")
    print("reports/phase1_4_rows_with_missing_values.csv")

    print("\nNo observations were automatically deleted.")
    print("No outliers were automatically removed.")
    print("No missing dates were filled.")
    print("\nPhase 1.4 audit completed.")


if __name__ == "__main__":
    main()