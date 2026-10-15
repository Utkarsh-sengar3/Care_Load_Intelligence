from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "uac_observed_forecasting_data.csv"
)

WEEKLY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "uac_weekly_forecasting_data.csv"
)

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_TARGET = "Children in HHS Care"

OTHER_TARGETS = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children discharged from HHS Care",
]


def main():
    print("\n" + "=" * 70)
    print("PHASE 2 - EXPLORATORY TIME-SERIES ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values("Date").reset_index(drop=True)

    weekly = pd.read_csv(
        WEEKLY_PATH,
        parse_dates=["Week_Start"],
    )

    print("\n1. DATASET OVERVIEW")
    print("-" * 70)

    print(f"Observed records: {len(df)}")
    print(f"Start date: {df['Date'].min().date()}")
    print(f"End date: {df['Date'].max().date()}")

    # --------------------------------------------------
    # 2. Primary target trend
    # --------------------------------------------------
    plt.figure(figsize=(15, 6))

    plt.plot(
        df["Date"],
        df[PRIMARY_TARGET],
        linewidth=1.3,
        label="Observed HHS Care Load",
    )

    plt.title("HHS Care Load Over Time")
    plt.xlabel("Date")
    plt.ylabel("Children in HHS Care")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase2_hhs_care_load_trend.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 3. Rolling mean and rolling standard deviation
    # --------------------------------------------------
    # These are windows of observed records, not necessarily
    # windows of consecutive calendar days.
    df["Rolling_Mean_14_Observations"] = (
        df[PRIMARY_TARGET]
        .rolling(window=14, min_periods=7)
        .mean()
    )

    df["Rolling_Std_14_Observations"] = (
        df[PRIMARY_TARGET]
        .rolling(window=14, min_periods=7)
        .std()
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        df["Date"],
        df[PRIMARY_TARGET],
        alpha=0.45,
        label="Observed care load",
    )

    plt.plot(
        df["Date"],
        df["Rolling_Mean_14_Observations"],
        linewidth=2,
        label="14-observation rolling mean",
    )

    plt.title("HHS Care Load and Rolling Mean")
    plt.xlabel("Date")
    plt.ylabel("Children in HHS Care")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase2_rolling_mean.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(15, 5))

    plt.plot(
        df["Date"],
        df["Rolling_Std_14_Observations"],
        label="14-observation rolling standard deviation",
    )

    plt.title("Rolling Variability in HHS Care Load")
    plt.xlabel("Date")
    plt.ylabel("Rolling Standard Deviation")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase2_rolling_variability.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 4. Changes between observed records
    # --------------------------------------------------
    df["Change_Between_Observations"] = (
        df[PRIMARY_TARGET].diff()
    )

    df["Absolute_Change"] = (
        df["Change_Between_Observations"].abs()
    )

    plt.figure(figsize=(15, 5))

    plt.plot(
        df["Date"],
        df["Change_Between_Observations"],
        linewidth=1,
        label="Change between observations",
    )

    plt.axhline(0, linestyle="--")

    plt.title("Changes in HHS Care Load Between Observations")
    plt.xlabel("Date")
    plt.ylabel("Change in Children in HHS Care")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase2_observation_changes.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 5. Compare target variables
    # --------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(15, 12),
    )

    columns = [PRIMARY_TARGET] + OTHER_TARGETS

    for ax, column in zip(axes.flatten(), columns):
        ax.plot(
            df["Date"],
            df[column],
            linewidth=1.1,
        )

        ax.set_title(column)
        ax.set_xlabel("Date")
        ax.set_ylabel("Recorded value")

    # Hide unused subplot.
    for ax in axes.flatten()[len(columns):]:
        ax.set_visible(False)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "phase2_all_variables.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # 6. Correlation heatmap
    # --------------------------------------------------
    correlation = df[columns].corr()

    fig, ax = plt.subplots(figsize=(11, 8))

    image = ax.imshow(
        correlation.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))

    ax.set_xticklabels(
        columns,
        rotation=60,
        ha="right",
    )

    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
            )

    ax.set_title("Correlation Between Recorded Variables")

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "phase2_correlation_heatmap.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # 7. Weekly coverage plot
    # --------------------------------------------------
    plt.figure(figsize=(15, 5))

    plt.plot(
        weekly["Week_Start"],
        weekly["Coverage_Percent"],
        marker=".",
        linewidth=1,
    )

    plt.axhline(
        100,
        linestyle="--",
        label="Complete coverage",
    )

    plt.axhline(
        50,
        linestyle=":",
        label="50% coverage",
    )

    plt.title("Weekly Observation Coverage")
    plt.xlabel("Week Start")
    plt.ylabel("Coverage (%)")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "phase2_weekly_coverage.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # 8. Statistical summary
    # --------------------------------------------------
    summary = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "Standard deviation",
            "Minimum",
            "Maximum",
            "First observed value",
            "Last observed value",
            "Overall change",
            "Mean absolute change",
            "Median absolute change",
        ],
        "Value": [
            df[PRIMARY_TARGET].mean(),
            df[PRIMARY_TARGET].median(),
            df[PRIMARY_TARGET].std(),
            df[PRIMARY_TARGET].min(),
            df[PRIMARY_TARGET].max(),
            df[PRIMARY_TARGET].iloc[0],
            df[PRIMARY_TARGET].iloc[-1],
            (
                df[PRIMARY_TARGET].iloc[-1]
                - df[PRIMARY_TARGET].iloc[0]
            ),
            df["Change_Between_Observations"].abs().mean(),
            df["Change_Between_Observations"].abs().median(),
        ],
    })

    summary["Value"] = summary["Value"].round(2)

    summary.to_csv(
        REPORTS_DIR / "phase2_hhs_care_load_summary.csv",
        index=False,
    )

    # Save correlation matrix.
    correlation.to_csv(
        REPORTS_DIR / "phase2_correlation_matrix.csv"
    )

    # Save data with rolling statistics.
    df.to_csv(
        REPORTS_DIR / "phase2_eda_observed_data.csv",
        index=False,
    )

    print("\n2. HHS CARE LOAD SUMMARY")
    print("-" * 70)
    print(summary.to_string(index=False))

    print("\n3. CORRELATION MATRIX")
    print("-" * 70)
    print(correlation.round(3).to_string())

    print("\n4. GENERATED FIGURES")
    print("-" * 70)

    for filename in [
        "phase2_hhs_care_load_trend.png",
        "phase2_rolling_mean.png",
        "phase2_rolling_variability.png",
        "phase2_observation_changes.png",
        "phase2_all_variables.png",
        "phase2_correlation_heatmap.png",
        "phase2_weekly_coverage.png",
    ]:
        print(f"reports/figures/{filename}")

    print("\nPhase 2 EDA completed.")


if __name__ == "__main__":
    main()