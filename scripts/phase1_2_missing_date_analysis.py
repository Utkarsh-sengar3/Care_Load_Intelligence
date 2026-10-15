import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_DATA_PATH


# Output directories
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("\n" + "=" * 65)
    print("PHASE 1.2 - MISSING DATE INVESTIGATION")
    print("=" * 65)

    # Load validated observed records
    data_path = ROOT / "data" / "interim" / "uac_observed_records.csv"

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Build calendar index for auditing only.
    # This does not mean missing values are zero.
    full_dates = pd.date_range(
        start=df["Date"].min(),
        end=df["Date"].max(),
        freq="D"
    )

    observed_dates = pd.DatetimeIndex(df["Date"].unique())
    missing_dates = full_dates.difference(observed_dates)

    print(f"\nObserved records: {len(df)}")
    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date: {df['Date'].max().date()}")
    print(f"Calendar days: {len(full_dates)}")
    print(f"Missing calendar dates: {len(missing_dates)}")

    # --------------------------------------------------
    # 1. Save missing dates with weekday information
    # --------------------------------------------------

    missing_df = pd.DataFrame({
        "Missing_Date": missing_dates
    })

    missing_df["Weekday"] = (
        missing_df["Missing_Date"].dt.day_name()
    )

    missing_df["Year"] = (
        missing_df["Missing_Date"].dt.year
    )

    missing_df["Month"] = (
        missing_df["Missing_Date"].dt.to_period("M").astype(str)
    )

    missing_df.to_csv(
        REPORTS_DIR / "missing_dates_detailed.csv",
        index=False
    )

    print("\nMissing dates by weekday:")
    print(
        missing_df["Weekday"]
        .value_counts()
        .reindex([
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ], fill_value=0)
        .to_string()
    )

    # --------------------------------------------------
    # 2. Monthly observation coverage
    # --------------------------------------------------

    calendar = pd.DataFrame({"Date": full_dates})

    observed_set = set(observed_dates)

    calendar["Observed"] = calendar["Date"].isin(observed_set)

    monthly = (
        calendar.set_index("Date")["Observed"]
        .resample("MS")
        .agg(["sum", "count"])
        .rename(columns={
            "sum": "Observed_Days",
            "count": "Calendar_Days"
        })
        .reset_index()
    )

    monthly["Missing_Days"] = (
        monthly["Calendar_Days"] - monthly["Observed_Days"]
    )

    monthly["Coverage_Percent"] = (
        monthly["Observed_Days"]
        / monthly["Calendar_Days"]
        * 100
    ).round(2)

    monthly.to_csv(
        REPORTS_DIR / "monthly_date_coverage.csv",
        index=False
    )

    print("\nMonthly coverage:")
    print(monthly.to_string(index=False))

    # --------------------------------------------------
    # 3. Gap intervals between observations
    # --------------------------------------------------

    df["Gap_Days"] = df["Date"].diff().dt.days

    gap_rows = df.loc[
        df["Gap_Days"] > 1,
        ["Date", "Gap_Days"]
    ].copy()

    gap_rows["Previous_Observed_Date"] = (
        gap_rows["Date"]
        - pd.to_timedelta(gap_rows["Gap_Days"], unit="D")
    )

    gap_rows = gap_rows[
        ["Previous_Observed_Date", "Date", "Gap_Days"]
    ]

    gap_rows.to_csv(
        REPORTS_DIR / "observation_gap_intervals.csv",
        index=False
    )

    print("\nLargest observation gaps:")
    print(
        gap_rows.sort_values(
            "Gap_Days", ascending=False
        ).head(15).to_string(index=False)
    )

    # --------------------------------------------------
    # 4. Plot monthly coverage
    # --------------------------------------------------

    plt.figure(figsize=(13, 5))

    plt.bar(
        monthly["Date"],
        monthly["Coverage_Percent"],
        width=20
    )

    plt.axhline(
        100,
        linestyle="--",
        label="Complete calendar coverage"
    )

    plt.title("Monthly Observation Coverage")
    plt.xlabel("Month")
    plt.ylabel("Observed Calendar Days (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "monthly_date_coverage.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 5. Plot missing dates by weekday
    # --------------------------------------------------

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_counts = (
        missing_df["Weekday"]
        .value_counts()
        .reindex(weekday_order, fill_value=0)
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        weekday_counts.index,
        weekday_counts.values
    )

    plt.title("Missing Dates by Day of Week")
    plt.xlabel("Weekday")
    plt.ylabel("Number of Missing Dates")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "missing_dates_by_weekday.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 6. Reporting frequency recommendation
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("PHASE 1.2 OUTPUTS")
    print("=" * 65)

    print("reports/missing_dates_detailed.csv")
    print("reports/monthly_date_coverage.csv")
    print("reports/observation_gap_intervals.csv")
    print("reports/figures/monthly_date_coverage.png")
    print("reports/figures/missing_dates_by_weekday.png")

    print("\nNo missing observations were imputed.")
    print("No missing dates were treated as zero activity.")
    print("\nPhase 1.2 completed.")


if __name__ == "__main__":
    main()