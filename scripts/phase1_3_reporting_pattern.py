from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "interim" / "uac_observed_records.csv"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "=" * 65)
    print("PHASE 1.3 - REPORTING PATTERN INVESTIGATION")
    print("=" * 65)

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    value_columns = [
        column for column in df.columns if column != "Date"
    ]

    # --------------------------------------------------
    # 1. Observation counts by weekday
    # --------------------------------------------------
    df["Weekday"] = df["Date"].dt.day_name()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday_counts = (
        df["Weekday"]
        .value_counts()
        .reindex(weekday_order, fill_value=0)
        .rename_axis("Weekday")
        .reset_index(name="Observed_Records")
    )

    weekday_counts["Share_Percent"] = (
        weekday_counts["Observed_Records"]
        / len(df)
        * 100
    ).round(2)

    weekday_counts.to_csv(
        REPORTS_DIR / "observed_records_by_weekday.csv",
        index=False,
    )

    print("\n1. OBSERVATIONS BY WEEKDAY")
    print(weekday_counts.to_string(index=False))

    # --------------------------------------------------
    # 2. Date-gap investigation
    # --------------------------------------------------
    df["Gap_Days"] = df["Date"].diff().dt.days

    gap_summary = (
        df["Gap_Days"]
        .dropna()
        .value_counts()
        .sort_index()
        .rename_axis("Gap_Days")
        .reset_index(name="Frequency")
    )

    gap_summary.to_csv(
        REPORTS_DIR / "observation_gap_frequency.csv",
        index=False,
    )

    print("\n2. OBSERVATION GAP FREQUENCY")
    print(gap_summary.to_string(index=False))

    # --------------------------------------------------
    # 3. Weekly aggregation and coverage
    # --------------------------------------------------
    df["Week_Start"] = (
        df["Date"]
        - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    ).dt.normalize()

    weekly = (
        df.groupby("Week_Start")
        .agg(
            Observed_Days=("Date", "nunique"),
            **{
                column: (column, "mean")
                for column in value_columns
            },
        )
        .reset_index()
    )

    # Each calendar week has 7 days, except partial
    # weeks at the beginning and end of the dataset.
    first_date = df["Date"].min()
    last_date = df["Date"].max()

    week_end = weekly["Week_Start"] + pd.Timedelta(days=6)

    weekly["Calendar_Days_In_Range"] = (
        (
            pd.concat(
                [
                    week_end,
                    pd.Series(
                        [last_date] * len(weekly),
                        index=weekly.index,
                    ),
                ],
                axis=1,
            ).min(axis=1)
            -
            pd.concat(
                [
                    weekly["Week_Start"],
                    pd.Series(
                        [first_date] * len(weekly),
                        index=weekly.index,
                    ),
                ],
                axis=1,
            ).max(axis=1)
        ).dt.days
        + 1
    )

    weekly["Coverage_Percent"] = (
        weekly["Observed_Days"]
        / weekly["Calendar_Days_In_Range"]
        * 100
    ).round(2)

    weekly.to_csv(
        REPORTS_DIR / "weekly_aggregated_data.csv",
        index=False,
    )

    print("\n3. WEEKLY DATASET SUMMARY")
    print(f"Number of calendar weeks: {len(weekly)}")
    print(
        "Weeks with 100% observation coverage:",
        int((weekly["Coverage_Percent"] == 100).sum()),
    )
    print(
        "Weeks with less than 50% coverage:",
        int((weekly["Coverage_Percent"] < 50).sum()),
    )
    print(
        "Mean weekly coverage:",
        round(weekly["Coverage_Percent"].mean(), 2),
        "%",
    )

    print("\nLowest-coverage weeks:")
    print(
        weekly[
            ["Week_Start", "Observed_Days",
             "Calendar_Days_In_Range", "Coverage_Percent"]
        ]
        .sort_values("Coverage_Percent")
        .head(15)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 4. Changes between observed records
    # --------------------------------------------------
    changes = df[["Date", "Gap_Days"]].copy()

    for column in value_columns:
        changes[f"{column}_Change"] = df[column].diff()

    changes.to_csv(
        REPORTS_DIR / "changes_between_observations.csv",
        index=False,
    )

    print("\n4. CHANGE ANALYSIS")
    print(
        "Change analysis saved to "
        "reports/changes_between_observations.csv"
    )

    print("\n" + "=" * 65)
    print("PHASE 1.3 COMPLETED")
    print("=" * 65)

    print("\nGenerated files:")
    print("reports/observed_records_by_weekday.csv")
    print("reports/observation_gap_frequency.csv")
    print("reports/weekly_aggregated_data.csv")
    print("reports/changes_between_observations.csv")


if __name__ == "__main__":
    main()