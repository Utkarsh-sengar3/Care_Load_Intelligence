import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_PATH
from src.data_validation import audit_dataframe, parse_numeric_columns


# Output directories
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("\n" + "=" * 60)
    print("PHASE 1.1 - DATA CLEANING AND VALIDATION")
    print("=" * 60)

    # 1. Load original dataset
    df = pd.read_csv(
        DEFAULT_DATA_PATH,
        skip_blank_lines=False
    )

    print(f"\nOriginal CSV rows: {len(df)}")

    # 2. Remove completely empty rows only
    empty_rows = df.isna().all(axis=1).sum()

    df = df.loc[~df.isna().all(axis=1)].copy()

    print(f"Completely empty rows removed: {empty_rows}")
    print(f"Rows after removing blank rows: {len(df)}")

    # 3. Convert dates and numerical columns
    df = parse_numeric_columns(
        df,
        date_column="Date"
    )

    # 4. Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # 5. Validate dates
    if df["Date"].isna().any():
        raise ValueError(
            "Invalid or missing dates found in populated records."
        )

    if df["Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected. Resolve them before continuing."
        )

    # 6. Check numerical columns
    numeric_columns = [
        col for col in df.columns
        if col != "Date"
    ]

    print("\nNumerical column validation:")

    for col in numeric_columns:

        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"Column is not numeric: {col}"
            )

        print(
            f"{col}: "
            f"missing={df[col].isna().sum()}, "
            f"negative={(df[col].dropna() < 0).sum()}"
        )

    # Do not silently fill missing values or delete negative values.
    # Investigate any unexpected values separately.

    # 7. Date-gap analysis
    unique_dates = pd.DatetimeIndex(
        df["Date"].drop_duplicates().sort_values()
    )

    full_calendar = pd.date_range(
        start=unique_dates.min(),
        end=unique_dates.max(),
        freq="D"
    )

    missing_dates = full_calendar.difference(unique_dates)

    missing_dates_df = pd.DataFrame({
        "Missing_Date": missing_dates
    })

    # 8. Consecutive observation gaps
    gaps = df["Date"].diff().dt.days.dropna()

    gap_summary = (
        gaps.value_counts()
        .sort_index()
        .rename_axis("Gap_Days")
        .reset_index(name="Occurrences")
    )

    # 9. Save interim dataset
    cleaned_path = (
        INTERIM_DIR / "uac_observed_records.csv"
    )

    df.to_csv(cleaned_path, index=False)

    # 10. Save missing dates
    missing_dates_path = (
        REPORTS_DIR / "missing_calendar_dates.csv"
    )

    missing_dates_df.to_csv(
        missing_dates_path,
        index=False
    )

    # 11. Save gap frequency
    gap_summary_path = (
        REPORTS_DIR / "date_gap_summary.csv"
    )

    gap_summary.to_csv(
        gap_summary_path,
        index=False
    )

    # 12. Create date-gap visualization
    plt.figure(figsize=(10, 5))

    plt.bar(
        gap_summary["Gap_Days"],
        gap_summary["Occurrences"]
    )

    plt.title(
        "Distribution of Gaps Between Observations"
    )

    plt.xlabel(
        "Days Between Consecutive Observations"
    )

    plt.ylabel(
        "Number of Occurrences"
    )

    plt.xticks(gap_summary["Gap_Days"])

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR / "date_gap_distribution.png"
    )

    plt.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # 13. Print final summary
    print("\n" + "=" * 60)
    print("PHASE 1.1 RESULTS")
    print("=" * 60)

    print(f"Validated records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date: {df['Date'].max().date()}")
    print(f"Unique dates: {df['Date'].nunique()}")
    print(f"Missing calendar dates: {len(missing_dates)}")

    print("\nDate-gap distribution:")
    print(gap_summary.to_string(index=False))

    print("\nFiles created:")
    print(cleaned_path)
    print(missing_dates_path)
    print(gap_summary_path)
    print(figure_path)

    print("\nPhase 1.1 completed.")


if __name__ == "__main__":
    main()