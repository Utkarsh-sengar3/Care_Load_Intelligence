# Care Load Intelligence — Machine Learning Notebooks Guide

This directory contains the complete educational, experimental, and reproducible Jupyter Notebook suite for **Care Load Intelligence: Predictive Forecasting of Care Load & Placement Demand**.

All notebooks are fully executed, self-contained, and grounded in the actual HHS Unaccompanied Alien Children operational dataset (2023–2025).

---

## Notebook Directory Structure

| Notebook File | Description | Execution Time | Output Size |
| :--- | :--- | :--- | :--- |
| **`Care_Load_Intelligence_End_to_End_ML.ipynb`** | **Master End-to-End Educational Notebook.** Covers all 23 stages from raw data auditing, sequence modeling, 4 time-series models, dual evaluation (walk-forward vs. holdout), empirical uncertainty, interval calibration, capacity stress-testing, to full dashboard mapping. | ~35s | ~1.93 MB |
| **`01_Data_Quality_and_EDA.ipynb`** | **Data Engineering & Exploratory Analysis.** In-depth inspection of 1,170 raw records vs. 720 valid observed records, analysis of 355 missing calendar days, gap distributions, rolling volatility, and operational correlation matrices. | ~5s | ~322 KB |
| **`02_Forecasting_Model_Experiments.ipynb`** | **Model Zoo & Holdout Experiments.** Stationarity testing (ADF test on levels vs. differences), chronological 80/20 train/test split (576/144 records), fitting Naive, ARIMA(1,1,1), Holt's Linear, and Holt's Damped Trend models. | ~8s | ~150 KB |
| **`03_Backtesting_and_Error_Analysis.ipynb`** | **Walk-Forward Validation & Horizon Analysis.** Rolling 45-origin backtesting (origin 400, step 7, horizon 7 = 315 forecasts), horizon sensitivity (steps 1–7), and residual diagnostic distributions. | ~15s | ~96 KB |
| **`04_Uncertainty_and_Capacity_Planning.ipynb`** | **Uncertainty Quantification & Capacity Planning.** Empirical error quantile calculation (Q80, Q90, Q95), split-origin calibration, final 7-step forecast, and capacity stress simulation (+10%, +25% shocks against 5,000 threshold). | ~6s | ~97 KB |

---

## Suggested Execution Order

If exploring modularly:
1. **`01_Data_Quality_and_EDA.ipynb`** — Understand data provenance, calendar gaps, and operational metrics.
2. **`02_Forecasting_Model_Experiments.ipynb`** — Explore baseline models and the 144-record fixed holdout.
3. **`03_Backtesting_and_Error_Analysis.ipynb`** — Run the 45-origin walk-forward evaluation and horizon sensitivity.
4. **`04_Uncertainty_and_Capacity_Planning.ipynb`** — Inspect empirical uncertainty bands and capacity stress testing.

Alternatively, execute **`Care_Load_Intelligence_End_to_End_ML.ipynb`** for the complete unified experience.

---

## Environment & Dependencies

The notebooks require Python 3.10+ with standard scientific libraries:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter nbconvert ipykernel openpyxl
```

Verified environment:
- Python: `3.10.0`
- pandas: `2.3.3`
- numpy: `2.2.6`
- matplotlib: `3.10.8`
- seaborn: `0.13.2`
- statsmodels: `0.15.0`
- scikit-learn: `1.7.2`

---

## How to Launch and Run the Notebooks

### Method 1: Using Jupyter Lab or Jupyter Notebook (Interactive GUI)
From the repository root:
```bash
jupyter lab
```
or
```bash
jupyter notebook
```
Navigate to the `notebooks/` folder and open any notebook. To execute all cells sequentially:
- Click **Kernel** $\rightarrow$ **Restart Kernel and Run All Cells...**

### Method 2: Command-Line Execution (Headless Batch Execution)
To re-run any notebook and update its embedded outputs in-place:
```bash
# Execute master end-to-end notebook
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Care_Load_Intelligence_End_to_End_ML.ipynb

# Execute supporting notebooks
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_Data_Quality_and_EDA.ipynb notebooks/02_Forecasting_Model_Experiments.ipynb notebooks/03_Backtesting_and_Error_Analysis.ipynb notebooks/04_Uncertainty_and_Capacity_Planning.ipynb
```

---

## How to Export Notebooks as Standalone HTML Reports

To generate standalone HTML reports suitable for offline viewing, academic submission, or sharing with non-technical stakeholders:

```bash
# Export master notebook to HTML
python -m jupyter nbconvert --to html notebooks/Care_Load_Intelligence_End_to_End_ML.ipynb --output Care_Load_Intelligence_ML_Report.html

# Export all notebooks to HTML
python -m jupyter nbconvert --to html notebooks/*.ipynb
```

The resulting HTML files are completely self-contained with embedded high-resolution charts and interactive tables.

---

## Methodological Highlights & Key Invariants

1. **Strict Chronological Splitting:** Random shuffling is strictly prohibited to prevent future-data leakage in time series.
2. **Zero-Imputation Prohibition:** Unobserved calendar days (weekends, holidays) are maintained as genuine reporting pauses and are never filled with zero.
3. **Dual Evaluation Distinction:** 
   - *Short-Horizon Rolling Walk-Forward (7 steps):* Holt's Damped Trend wins (MAE = 78.92 vs. Naive = 102.61).
   - *Long-Horizon Single Holdout (144 steps):* Naive wins (MAE = 186.51 vs. Holt Damped = 211.91; Holt Linear = 703.51).
4. **Empirical Uncertainty:** Bounds are derived directly from the distribution of historical absolute forecast errors, avoiding unrealistic Gaussian assumptions.
5. **Illustrative Planning Threshold:** The 5,000 capacity figure is a configurable operational planning assumption, not certified official shelter capacity.
