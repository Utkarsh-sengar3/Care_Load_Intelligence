# Care Load Intelligence — Professional Video Presentation Script

**Project Title:** Care Load Intelligence: Predictive Forecasting of Care Load & Placement Demand  
**Live Application:** [https://cli-utkaresh-sengar-updated.streamlit.app/](https://cli-utkaresh-sengar-updated.streamlit.app/)  
**GitHub Repository:** [https://github.com/Utkarsh-sengar3/Care_Load_Intelligence_Updated_Website](https://github.com/Utkarsh-sengar3/Care_Load_Intelligence_Updated_Website)  
**Presenter / Developer:** Utkarsh Sengar  
**Tone:** Technical, authoritative, natural, measured, highly professional  

---

## Navigation & Script Index
1. [Version 1: Full Technical Presentation (5–7 Minutes)](#version-1-full-technical-presentation-57-minutes)
2. [Version 2: Concise Executive Demonstration (2 Minutes)](#version-2-concise-executive-demonstration-2-minutes)
3. [Version 3: High-Impact Portfolio Pitch (60 Seconds)](#version-3-high-impact-portfolio-pitch-60-seconds)

---

## Version 1: Full Technical Presentation (5–7 Minutes)

### Target Timing Breakdown
- **Part 1 — Introduction:** 0:00 – 0:35 (35s)
- **Part 2 — Problem Statement & Domain Context:** 0:35 – 1:15 (40s)
- **Part 3 — Dataset Provenance & Irregularity:** 1:15 – 2:05 (50s)
- **Part 4 — Machine Learning & Forecasting Workflow:** 2:05 – 3:05 (60s)
- **Part 5 — Model Evaluation & The Horizon Dilemma:** 3:05 – 3:55 (50s)
- **Part 6 — Live Website Interactive Walkthrough:** 3:55 – 5:35 (100s)
- **Part 7 — Empirical Uncertainty & Operational Capacity:** 5:35 – 6:20 (45s)
- **Part 8 — Conclusion & Key Takeaways:** 6:20 – 6:50 (30s)
- **Total Duration:** ~6 minutes 50 seconds

---

### Part 1 — Introduction (0:00 – 0:35)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:15** | **Title Card & Presenter On-Camera**<br>Display project title: *Care Load Intelligence: Predictive Forecasting of Care Load & Placement Demand* with presenter name *Utkarsh Sengar* and GitHub badge. | "Hello everyone. My name is Utkarsh Sengar, and today I am presenting **Care Load Intelligence** — an applied time-series machine learning system and decision-support dashboard designed for predictive forecasting of care load and placement demand." | Fade in from black; crisp lower-third graphic. |
| **0:15 – 0:35** | **Architecture High-Level Graphic**<br>Cut to `06_ml_workflow_architecture.png` highlighting the path from raw operational records to capacity simulation. | "Managing humanitarian and transitional care systems requires understanding historical population dynamics, anticipating demand surges, and systematically evaluating capacity stress. In this project, I engineered an end-to-end forecasting pipeline that moves from irregular operational observations to rigorous model evaluation, empirical uncertainty estimation, and an interactive production dashboard." | Smooth slide transition to workflow graphic. |

---

### Part 2 — Problem Statement & Domain Context (0:35 – 1:15)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **0:35 – 0:55** | **Operational Context Graphic**<br>Display HHS UAC shelter care context and historical care load curve (`01_historical_care_load_trend.png`). | "In temporary custodial and care programs — specifically inspired by the U.S. Department of Health and Human Services Unaccompanied Alien Children operational reports — shelter managers face volatile inflows, complex placement logistics, and severe operational constraints." | Pan slowly across the historical care load wave. |
| **0:55 – 1:15** | **The Forecasting Need**<br>Highlighting uncertainty and decision support: text callouts: *Proactive Bed Planning*, *Empirical Risk Bounds*, *Defensible Model Selection*. | "Care networks cannot afford reactive adjustments. A sudden surge can strain facilities, while an unexpected lull leaves staff underutilized. However, off-the-shelf forecasting often fails here because operational data is messy, non-stationary, and irregularly reported. The goal of this project was not to claim impossible oracle predictions, but to build a transparent, statistically defensible decision-support system that quantifies both expected load and empirical uncertainty." | Quick zoom on the callout pillars. |

---

### Part 3 — Dataset Provenance & Irregularity (1:15 – 2:05)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **1:15 – 1:40** | **Data Audit & Structure**<br>Screen recording of Jupyter notebook or `data/processed/uac_observed_forecasting_data.csv` table. Show columns: `Date`, `Children in CBP custody`, `Children in HHS Care` (Target). | "Let’s look at the actual data. The dataset spans from January 12th, 2023 through December 21st, 2025. While the calendar span is 1,075 days, there are exactly 720 valid observed records. The raw dataset contained 1,170 rows, but 450 of those were completely blank summary rows and footnotes that required automated data validation." | Highlight raw vs. processed table shape (720 rows). |
| **1:40 – 2:05** | **Date Gap & Irregularity Analysis**<br>Display `reports/figures/date_gap_distribution.png` showing 558 one-day gaps, 122 three-day gaps, and 23 four-day gaps. | "Here is a crucial methodological insight: 355 calendar days are absent from reporting. 558 intervals are 1-day consecutive reports, while 122 are 3-day gaps corresponding to weekend reporting pauses. In operational time series, **missing reporting dates must never be imputed as zero care load**. Treating an unobserved weekend as zero children would destroy model training. Instead, we formulated the forecasting problem over consecutive observed records and parallel weekly aggregations." | Pop-up warning badge: *Never Impute Missing Operational Dates as Zero*. |

---

### Part 4 — Machine Learning & Forecasting Workflow (2:05 – 3:05)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **2:05 – 2:30** | **Model Zoo Architecture**<br>Cut to diagram showing Naive Baseline, ARIMA(1,1,1), Holt's Linear, and Holt's Damped Trend. | "To forecast the target variable — *Children in HHS Care* — I implemented and benchmarked multiple time-series paradigms: first, a Naive persistence benchmark, carrying forward the last known state; second, an autoregressive integrated moving average model, specifically ARIMA(1,1,1), capturing first-order momentum and differencing; and third, exponential smoothing architectures including Holt's Linear Trend and Holt's Damped Trend." | Highlight each model icon as it is introduced. |
| **2:30 – 3:05** | **Validation Design: Why Damped Trend Matters**<br>Show equations or parameter cards: Level ($l_t$), Trend ($b_t$), Damping parameter ($\phi < 1$). | "In non-stationary human caseload series, linear trends frequently overshoot when extrapolated forward. Holt's Damped Trend introduces a damping parameter, phi, which gradually attenuates the projected slope into the future, preventing runaway overshoots. Furthermore, we avoided data leakage by maintaining strict chronological ordering — never using random splits for time-dependent series." | Visual representation of linear trend exploding vs damped trend flattening. |

---

### Part 5 — Model Evaluation & The Horizon Dilemma (3:05 – 3:55)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **3:05 – 3:30** | **Walk-Forward Validation Chart**<br>Display `02_model_performance_comparison.png` (left panel) and `07_actual_vs_predicted_scatter.png`. | "Now, how did the models perform? We tested them under two distinct evaluation setups. In our primary walk-forward validation — consisting of 45 rolling origins with an initial training window of 400 records and a 7-step forecast horizon — **Holt Damped Trend achieved the top performance**, with a Mean Absolute Error of 78.9 children, an RMSE of 160.8, and a MAPE under 2.0%." | Highlight Holt Damped: MAE = 78.9 vs Naive = 102.6. |
| **3:30 – 3:55** | **The Chronological Holdout Contrast**<br>Display `02_model_performance_comparison.png` (right panel) comparing the 144-record holdout results. | "However, a senior data scientist must present the complete picture. When evaluated on a single-origin 144-record holdout without intermediate retraining, Naive and ARIMA(1,1,1) scored an MAE of 186.5, outperforming Holt Damped at 211.9, while Holt Linear completely broke down at 703.5. This proves a vital principle: model rankings depend heavily on the evaluation horizon and retraining cadence. Holt Damped excels for tactical 7-step rolling forecasts, whereas simple persistence remains remarkably robust over un-updated long horizons." | Split-screen showing short-horizon winner vs long-horizon benchmark. |

---

### Part 6 — Live Website Interactive Walkthrough (3:55 – 5:35)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **3:55 – 4:15** | **Live Website: Overview & Forecast Controls**<br>Capture screen of `https://cli-utkaresh-sengar-updated.streamlit.app/`. Show sidebar: Select `Holt_Damped`, horizon `7`, capacity `5000`, warning `80%`. | "Now let's explore the live application deployed on Streamlit Cloud. The workspace features 12 dedicated analytical modules with custom dark and light themes. In the sidebar, the user can select their forecasting model, set the projection horizon in terms of future observed records, configure scenario capacity, and tune warning thresholds." | Switch browser tab to live URL; cursor moves to sidebar. |
| **4:15 – 4:35** | **Overview & Forecast Explorer Pages**<br>Navigate to *Overview*, hover on KPI metric cards, scroll to interactive Plotly chart. Then click *Forecast explorer*. | "On the **Overview** page, we see immediate executive KPIs: the latest observed load of 2,484 children, the historical average of 3,554, and the 7-step forecast trajectory. Switching to the **Forecast Explorer**, we can inspect the forecast alongside empirical 90% error bands, with interactive zoom and tooltips directly illustrating where future demand tracks relative to warning thresholds." | Hover cursor over forecast line and shaded Q90 band. |
| **4:35 – 4:55** | **Model Evaluation & Backtesting Simulator**<br>Click *Model evaluation* radio, show comparison table and actual vs predicted plot. Then click *Backtesting simulator*. | "In the **Model Evaluation** page, stakeholders can toggle between walk-forward cross-validation metrics and chronological holdout splits. The **Backtesting Simulator** allows analysts to replay historical prediction origins step-by-step, auditing where individual model residuals spiked during past inflection points." | Click through tabs and scroll through metrics table. |
| **4:55 – 5:15** | **Capacity Planner & Scenario Lab**<br>Navigate to *Capacity planner*, show utilization gauges. Then navigate to *Scenario lab*, toggle scenario checkboxes. | "Next, the **Capacity Planner** evaluates shelter demand against capacity. With our current 2,484 load and a 5,000 bed threshold, current utilization is roughly 50%, designated as 'Low Risk'. In the **Scenario Lab**, we test hypothetical stress multipliers — a 10% elevated demand shock and a 25% severe surge scenario. Even under a 25% shock, projected peak demand reaches 3,125 children, remaining safely below the 4,000-bed warning line." | Toggle checkboxes: `Baseline`, `Elevated_Demand`, `High_Demand_Stress`. |
| **5:15 – 5:35** | **Data Quality & Report Center**<br>Navigate to *Data quality*, then *Report center*. Click **Download PDF report** and **Download Excel workbook**. | "Finally, the **Data Quality** page transparently communicates date coverage and gap distributions, while the **Report Center** enables single-click export of executive PDF briefs generated dynamically with ReportLab and multi-sheet Excel workbooks." | Click download button; show downloaded file icon briefly. |

---

### Part 7 — Empirical Uncertainty & Operational Capacity (5:35 – 6:20)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **5:35 – 6:00** | **Empirical Uncertainty vs Theory**<br>Display `03_forecast_uncertainty_intervals.png` and interval calibration table from Phase 3.5 / 3.6. | "A critical pillar of this project is responsible uncertainty communication. Rather than assuming Gaussian residuals that rarely hold in real operations, we derived empirical error quantiles — 80%, 90%, and 95% — from 315 out-of-sample backtest errors. On evaluation, the uncalibrated 90% interval achieved 86% empirical coverage, while our chronological calibration phase safely encompassed 100% of evaluation points." | Show empirical quantile formulas and interval widths. |
| **6:00 – 6:20** | **Capacity Assumption Disclosure**<br>Display disclaimer banner on screen. | "I must also emphasize an essential operational disclaimer: the 5,000 capacity threshold used throughout the dashboard is an illustrative planning benchmark, not an officially certified facility ceiling. In a live deployment, this input is dynamically updated with facility-specific operational bed counts." | Highlight transparent disclosure badge. |

---

### Part 8 — Conclusion & Key Takeaways (6:20 – 6:50)

| Timestamp | Scene / Visual | Spoken Narration | Transition / Action |
| :--- | :--- | :--- | :--- |
| **6:20 – 6:35** | **Summary Grid**<br>Display four key pillars: *Sound Data Engineering*, *Empirical Horizon-Aware ML*, *Transparent Uncertainty*, *Production-Ready Analytics*. | "To summarize, Care Load Intelligence provides an end-to-end framework combining rigorous data cleaning, multi-horizon time-series evaluation, empirical uncertainty quantification, and an interactive decision-support application." | Icons illuminate sequentially on screen. |
| **6:35 – 6:50** | **Closing Card & Contact Info**<br>Display live demo link, GitHub repo link, LinkedIn profile, and presenter name: *Utkarsh Sengar*. | "All source code, automated test suites, precomputed reports, and educational Jupyter notebooks are publicly available on my GitHub repository, and you can test the live application at the link below. Thank you very much for your time, and I welcome your questions and feedback!" | Fade out to closing title card with social links. |

---

## Version 2: Concise Executive Demonstration (2 Minutes)

**Target Duration:** 2 minutes (approx. 270 words)  
**Audience:** Hiring managers, tech leads, executive stakeholders  

> "Hi, I’m Utkarsh Sengar. Welcome to a two-minute walkthrough of **Care Load Intelligence**, an applied time-series forecasting and capacity planning platform.
>
> In custodial care and humanitarian placement programs, shelter managers face volatile demand patterns and irregular operational reporting. Using historical Department of Health and Human Services data spanning from January 2023 to December 2025, I developed an end-to-end analytics platform to forecast placement demand and stress-test shelter capacity.
>
> The raw data contains 720 observed records across 1,075 calendar days. Because reporting pauses on weekends and holidays, missing dates cannot simply be imputed with zero. Our pipeline treats the series as an irregular sequence of observed records and parallel weekly aggregations.
>
> We evaluated Naive persistence, ARIMA(1,1,1), and Holt's Exponential Smoothing models. In our 45-origin rolling walk-forward evaluation, **Holt's Damped Trend model delivered the best short-horizon accuracy**, achieving a Mean Absolute Error of 78.9 children — outperforming both ARIMA and Naive baselines. Damping the trend prevented explosive over-projections while adapting to local shifts.
>
> On the live Streamlit dashboard, decision-makers can select forecasting models, adjust horizons from 3 to 30 records, and inspect empirical 80%, 90%, and 95% uncertainty intervals computed from historical residual quantiles.
>
> The dashboard also features an interactive **Scenario Lab**, enabling planners to test 10% and 25% demand surge assumptions against configurable capacity thresholds, with one-click export of executive PDF and Excel reports.
>
> The complete codebase, reproducible Jupyter notebooks, and live application are available on my GitHub. Thank you!"

---

## Version 3: High-Impact Portfolio Pitch (60 Seconds)

**Target Duration:** 60 seconds (approx. 140 words)  
**Audience:** LinkedIn feed, recruiter portfolio screen, quick project showcase  

> "Can machine learning help humanitarian care facilities anticipate demand spikes before capacity bottlenecks occur?
>
> I built **Care Load Intelligence** — an end-to-end forecasting platform analyzing 720 historical operational records from 2023 through 2025.
>
> Instead of naïve imputation that misinterprets irregular weekend gaps as zero caseload, I engineered a sequence-aware forecasting pipeline.
>
> In a 45-origin rolling walk-forward backtest, **Holt's Damped Trend achieved top performance with an MAE of 78.9 children**, successfully capturing momentum without the runaway trend errors of linear models.
>
> The deployed Streamlit web application features 12 analytical pages, empirical uncertainty intervals, demand stress-testing for 10% and 25% surge scenarios, and automated PDF report generation.
>
> Explore the live interactive dashboard and GitHub repository in the links below. I’m Utkarsh Sengar — thanks for watching!"
