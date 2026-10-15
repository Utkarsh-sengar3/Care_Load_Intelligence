# Care Load Intelligence — Updated Dashboard

## What's included
- Restrained enterprise analytics UI inspired by professional Streamlit / healthcare operations dashboards.
- Dark-first theme with a light report-view option.
- Overview and forecast explorer with observed history, forecast trajectory, capacity reference, and empirical error ranges where available.
- Model evaluation with separate walk-forward and final holdout views plus prediction-error inspection.
- Historical backtesting simulator: choose a chronological cutoff, fit only on earlier observations, and compare with subsequent observed records.
- Forecast uncertainty page with explicit limitations.
- Capacity planner and hypothetical demand scenarios.
- Horizon sensitivity comparison for 3, 7, 14, and 30 future observed records.
- Data quality monitor for reporting gaps, invalid dates, duplicate dates, missing target values, interval distribution, and weekday pattern.
- Report center exporting a PDF summary, Excel workbook, and CSV snapshots.
- Data explorer and methodology/limitations pages.

## Important scientific safeguards
- The dataset has irregular observation dates. Forecast horizons mean **future observed records**, not calendar days.
- Missing calendar dates are not converted to zero.
- Capacity is an editable planning assumption (default 5,000), not a verified official capacity.
- +10% / +25% demand scenarios are hypothetical stress assumptions, not probabilistic predictions.
- Empirical historical error ranges are not guaranteed prediction intervals and may not achieve nominal coverage.
- Backtesting is a single-origin replay; interpret it alongside the existing walk-forward and chronological holdout reports.

## Install and run (Windows PowerShell)
1. Extract this ZIP.
2. Open the `uac-care-forecasting` folder in VS Code.
3. Create/activate a virtual environment (skip creation if you already have a working `.venv`):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Run the app:

   ```powershell
   python -m streamlit run app.py
   ```

6. Open `http://localhost:8501` if the browser does not open automatically.

## Files and folders
- `app.py` — updated Streamlit website.
- `src/forecasting_engine.py` — preserved forecasting engine.
- `data/processed/uac_observed_forecasting_data.csv` — preserved processed dataset.
- `reports/` — preserved evaluation, prediction, and supporting report artifacts.
- `requirements.txt` — dependencies including Excel and PDF report export support.

Do not replace `src/forecasting_engine.py`, the processed CSV, or the reports folder when applying only the UI update to an existing copy. Back up your current `app.py` before replacing it.

## Validation status
The updated `app.py` was syntax-compiled with Python. A live Streamlit browser test was not run in this build environment, so run it locally and share any traceback if a runtime issue appears.
