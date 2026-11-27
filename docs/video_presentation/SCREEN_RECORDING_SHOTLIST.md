# Care Load Intelligence — Screen Recording Shot List

**Target Resolution:** 1920 × 1080 (1080p, 60fps or 30fps)  
**Browser:** Chrome or Brave, maximized window, zoom 100% (or 110% for crisp typography)  
**Live URL:** [https://cli-utkaresh-sengar-updated.streamlit.app/](https://cli-utkaresh-sengar-updated.streamlit.app/)  
**Presenter:** Utkarsh Sengar  

---

## Shot List Overview

| Shot ID | Time in Video | Asset / Page | Action / Interaction Description | Visual Focus |
| :--- | :--- | :--- | :--- | :--- |
| **SHOT-01** | 0:00 – 0:15 | Title Slide / Webcam | Presenter on camera or crisp title slide with name & GitHub icon. | Title, Presenter Name |
| **SHOT-02** | 0:15 – 0:35 | `06_ml_workflow_architecture.png` | Pan from Data Ingestion (left) across to ML Zoo and Dashboard (right). | Full pipeline architecture |
| **SHOT-03** | 0:35 – 0:55 | `01_historical_care_load_trend.png` | Slow pan across the 2023–2025 care load wave. Highlight peak (8,765) and recent stabilization. | Cyclical & regime shifts |
| **SHOT-04** | 0:55 – 1:15 | Slide: Pillars of Care Planning | Highlight 3 cards: Proactive Bed Planning, Empirical Uncertainty, Defensible ML. | Core value propositions |
| **SHOT-05** | 1:15 – 1:40 | Jupyter Notebook / Processed Data | Scroll through DataFrame in notebook showing Date, CBP custody, and HHS Care columns. | 720 rows, 6 columns |
| **SHOT-06** | 1:40 – 2:05 | `reports/figures/date_gap_distribution.png` | Zoom in on gap histogram (558 1-day gaps, 122 weekend 3-day gaps). Highlight "Do not impute zero" callout. | Irregular reporting pattern |
| **SHOT-07** | 2:05 – 2:30 | Slide: Time-Series Zoo | Show cards for Naive, ARIMA(1,1,1), Holt's Linear, and Holt's Damped Trend. | Model equations & concepts |
| **SHOT-08** | 2:30 – 3:05 | Slide: Damped Trend Mechanism | Animate or contrast linear trend overshooting vs damped trend flattening out asymptotically. | Damping parameter $\phi < 1$ |
| **SHOT-09** | 3:05 – 3:30 | `02_model_performance_comparison.png` (Left) | Zoom in on Walk-Forward MAE bar chart. Highlight Holt Damped = 78.9 vs Naive = 102.6. | Walk-forward superiority |
| **SHOT-10** | 3:30 – 3:55 | `02_model_performance_comparison.png` (Right) | Zoom in on 144-record holdout bar chart. Point out Naive = 186.5 vs Holt Damped = 211.9 and Holt Linear = 703.5. | Horizon nuance |
| **SHOT-11** | 3:55 – 4:15 | Live Website: Sidebar Setup | Hover over sidebar logo. Change model to `Holt_Damped`, horizon slider to `7`, capacity to `5000`, warning to `80%`. | Sidebar controls |
| **SHOT-12** | 4:15 – 4:35 | Live Website: Overview & Forecast | Hover over 4 KPI cards (Latest: 2,484; Mean: 3,554). Hover over Plotly forecast line showing tooltips. | Executive summary metrics |
| **SHOT-13** | 4:35 – 4:55 | Live Website: Model Eval & Backtesting | Click radio to *Model evaluation*. Scroll through comparison table. Click *Backtesting simulator*, show origin replay slider. | Reproducible evaluation |
| **SHOT-14** | 4:55 – 5:15 | Live Website: Capacity & Scenario Lab | Click *Capacity planner*, show utilization metric (~49.7%). Click *Scenario lab*, check/uncheck +10% and +25% scenarios. | Stress testing and headroom |
| **SHOT-15** | 5:15 – 5:35 | Live Website: Report Center | Click *Report center*. Click "Download PDF report" and "Download Excel workbook". Show downloaded files. | Automated reporting |
| **SHOT-16** | 5:35 – 6:00 | `03_forecast_uncertainty_intervals.png` | Highlight empirical quantile bands (80%, 90%, 95%). Explain why they widen with horizon. | Empirical bounds vs Gaussian |
| **SHOT-17** | 6:00 – 6:20 | Slide / UI Disclaimer | Zoom in on disclaimer box: "Capacity of 5,000 is an illustrative planning assumption". | Transparency & governance |
| **SHOT-18** | 6:20 – 6:50 | Closing Summary & Links | Presenter webcam or closing slide with live demo link, GitHub repo link, and LinkedIn handle. | Call to action |

---

## Detailed Live Demonstration Instructions (Page-by-Page)

### 1. Preparation
1. Open Chrome and navigate to: `https://cli-utkaresh-sengar-updated.streamlit.app/`.
2. Ensure the Streamlit top menu is clean (you can collapse the hamburger menu if desired).
3. Set browser zoom to 100% or 110%.
4. Ensure sidebar theme is set to **"Dark — analyst workspace"**.

### 2. Live Sequence Choreography

#### Sidebar Controls (03:55 – 04:15)
- **Mouse Action:** Move cursor to left sidebar.
- **Model Dropdown:** Click "Forecast model" dropdown and verify `Holt_Damped` is selected.
- **Horizon Slider:** Click slider and verify `7` future observed records is selected.
- **Scenario Capacity:** Verify `5000` is typed in.
- **Warning Threshold:** Verify slider is at `80%`.

#### Page: Overview (04:15 – 04:25)
- **Mouse Action:** Click "Overview" in sidebar radio.
- **KPI Cards:** Move cursor slowly across the 4 top cards:
  - *Latest observed load*: 2,484
  - *Historical mean*: 3,554
  - *Historical maximum*: 8,765
  - *Mean forecast — 7 records*: ~2,494
- **Main Chart:** Hover over the historical line curve to show the interactive tooltip displaying date and load.
- **Forecast Trajectory Chart:** Scroll down slightly to show the point forecast line and empirical Q90 shaded area.

#### Page: Forecast Explorer (04:25 – 04:35)
- **Mouse Action:** Click "Forecast explorer" on sidebar.
- **Top Metrics:** Point out *Forecast range: 2,487 to 2,500*.
- **Chart:** Hover over step 1 through 7. Point out the horizontal dashed yellow line at 5,000 capacity and cyan dotted line at 4,000 warning threshold.
- **Table:** Scroll down to the forecast values table showing Lower Q90, Upper Q90, and Low Scenario Risk.

#### Page: Model Evaluation & Backtesting (04:35 – 04:55)
- **Mouse Action:** Click "Model evaluation" on sidebar.
- **View:** Scroll through the Walk-Forward Validation table highlighting Holt Damped's MAE of 78.92 and RMSE of 160.87.
- **Chart:** Show the Model Comparison bar chart.
- **Next Page:** Click "Backtesting simulator".
- **Interaction:** Move the historical evaluation origin slider to demonstrate how past forecasts tracked actual observed values.

#### Page: Capacity Planner & Scenario Lab (04:55 – 05:15)
- **Mouse Action:** Click "Capacity planner".
- **Metrics:** Highlight *Capacity Utilization*: ~49.7% (Low Risk).
- **Next Page:** Click "Scenario lab".
- **Interaction:** Uncheck "High_Demand_Stress" and re-check it to demonstrate live reactivity.
- **Chart:** Point out the three trajectory lines: Baseline (1.00x), Elevated Demand (1.10x), and High-Demand Stress (1.25x). Show that even at peak stress (3,125 children), it remains below the 4,000 warning line.

#### Page: Data Quality & Report Center (05:15 – 05:35)
- **Mouse Action:** Click "Data quality" to show date coverage and gap statistics.
- **Next Page:** Click "Report center".
- **Download Actions:**
  - Click **Download PDF report**. (ReportLab landscape document downloads instantly).
  - Click **Download Excel workbook**. (Multi-tab openpyxl workbook downloads instantly).
- **Smooth Transition:** Cut back to presenter camera or summary slides.

---

## Audio-Visual Synchronization Guide

- When narration mentions **"720 valid observed records"**, ensure SHOT-05 (Notebook or Data Explorer) is active on screen.
- When narration mentions **"Holt Damped Trend achieved an MAE of 78.9"**, cursor should hover over the 78.9 bar in SHOT-09.
- When narration mentions **"missing dates must never be imputed as zero"**, display the bold popup badge in SHOT-06.
- When narration mentions **"5,000 capacity threshold is illustrative"**, highlight the disclaimer banner in SHOT-17.
