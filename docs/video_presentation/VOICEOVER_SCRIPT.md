# Care Load Intelligence — Professional Voiceover Script

**Project Title:** Care Load Intelligence: Predictive Forecasting of Care Load & Placement Demand  
**Presenter:** Utkarsh Sengar  
**Recording Target:** Studio microphone or high-quality USB condenser (e.g., Shure MV7 / Rode NT-USB)  
**Pacing:** Confident, conversational, measured (approx. 130–145 words per minute)  

---

## Voiceover Recording Notes & Pronunciation Guide

| Term | Pronunciation Guide | Meaning / Notes |
| :--- | :--- | :--- |
| **Utkarsh Sengar** | *OOT-karsh SEN-gahr* | Presenter Name |
| **HHS** | *Aych-Aych-Ess* | Health and Human Services |
| **CBP** | *See-Bee-Pee* | Customs and Border Protection |
| **UAC** | *You-Ay-See* | Unaccompanied Alien Children |
| **ARIMA** | *uh-REE-muh* | AutoRegressive Integrated Moving Average |
| **Holt's Damped** | *Holts Dampt* | Exponential smoothing with trend damping |
| **MAE** | *Em-Ay-Ee* | Mean Absolute Error |
| **RMSE** | *Root-Mean-Squared-Error* or *R-M-S-E* | Root Mean Squared Error |
| **MAPE** | *MAY-pee* or *Em-Ay-Pee-Ee* | Mean Absolute Percentage Error |
| **Phi ($\phi$)** | *Fye* | Damping parameter ($0 < \phi < 1$) |

---

## 1. Master Voiceover Script (5–7 Minute Presentation)

### [PART 1 — INTRODUCTION | 0:00 – 0:35]
*(Tone: Engaging, authoritative, welcoming)*

"Hello everyone. My name is Utkarsh Sengar [PAUSE], and today I am presenting **Care Load Intelligence** — an applied time-series machine learning system and decision-support dashboard designed for predictive forecasting of care load and placement demand. [BREATH]

Managing humanitarian and transitional care systems requires understanding historical population dynamics, anticipating demand surges, and systematically evaluating capacity stress. In this project, I engineered an end-to-end forecasting pipeline that moves from irregular operational observations to rigorous model evaluation, empirical uncertainty estimation, and an interactive production dashboard."

---

### [PART 2 — PROBLEM STATEMENT & DOMAIN CONTEXT | 0:35 – 1:15]
*(Tone: Serious, analytical)*

"In temporary custodial and care programs — specifically inspired by the U.S. Department of Health and Human Services Unaccompanied Alien Children operational reports — shelter managers face volatile inflows, complex placement logistics, and severe operational constraints. [PAUSE]

Care networks cannot afford reactive adjustments. A sudden surge can strain facilities, while an unexpected lull leaves staff underutilized. However, off-the-shelf forecasting often fails here because operational data is messy, non-stationary, and irregularly reported. 

The goal of this project was not to claim impossible oracle predictions, but to build a transparent, statistically defensible decision-support system that quantifies both expected load and empirical uncertainty."

---

### [PART 3 — DATASET PROVENANCE & IRREGULARITY | 1:15 – 2:05]
*(Tone: Methodical, instructional)*

"Let’s look at the actual data. [PAUSE] 

The dataset spans from January 12th, 2023 through December 21st, 2025. While the calendar span covers 1,075 days, there are exactly 720 valid observed records. The raw dataset contained 1,170 rows, but 450 of those were completely blank summary rows and footnotes that required automated data validation. [BREATH]

Here is a crucial methodological insight: 355 calendar days are absent from reporting. 558 intervals represent 1-day consecutive weekday reports, while 122 are 3-day gaps corresponding to weekend reporting pauses. 

In operational time series, **missing reporting dates must never be imputed as zero care load**. [EMPHASIZE] Treating an unobserved weekend as zero children would destroy model training. Instead, we formulated the forecasting problem over consecutive observed records and parallel weekly aggregations."

---

### [PART 4 — MACHINE LEARNING & FORECASTING WORKFLOW | 2:05 – 3:05]
*(Tone: Technical, clear, confident)*

"To forecast the target variable — *Children in HHS Care* — I implemented and benchmarked multiple time-series paradigms: [PAUSE]

First, a Naive persistence benchmark, carrying forward the last known state. 

Second, an autoregressive integrated moving average model, specifically ARIMA(1,1,1), capturing first-order momentum and differencing. 

And third, exponential smoothing architectures including Holt's Linear Trend and Holt's Damped Trend. [BREATH]

In non-stationary human caseload series, linear trends frequently overshoot when extrapolated forward. Holt's Damped Trend introduces a damping parameter, phi, which gradually attenuates the projected slope into the future, preventing runaway overshoots. 

Furthermore, we avoided future-data leakage by maintaining strict chronological ordering — never using random splits for time-dependent series."

---

### [PART 5 — MODEL EVALUATION & THE HORIZON DILEMMA | 3:05 – 3:55]
*(Tone: Analytical, nuanced, intellectual)*

"Now, how did the models perform? We tested them under two distinct evaluation setups. [PAUSE]

In our primary walk-forward validation — consisting of 45 rolling origins with an initial training window of 400 records and a 7-step forecast horizon — **Holt Damped Trend achieved the top performance**, with a Mean Absolute Error of 78.9 children, an RMSE of 160.8, and a MAPE under 2.0%. [BREATH]

However, a senior data scientist must present the complete picture. [SLOW DOWN]

When evaluated on a single-origin 144-record holdout without intermediate retraining, Naive and ARIMA(1,1,1) scored an MAE of 186.5, outperforming Holt Damped at 211.9, while Holt Linear completely broke down at 703.5. 

This proves a vital principle: model rankings depend heavily on the evaluation horizon and retraining cadence. Holt Damped excels for tactical 7-step rolling forecasts, whereas simple persistence remains remarkably robust over un-updated long horizons."

---

### [PART 6 — LIVE WEBSITE INTERACTIVE WALKTHROUGH | 3:55 – 5:35]
*(Tone: Upbeat, demonstrative, guided)*

"Now let's explore the live application deployed on Streamlit Cloud. [PAUSE]

The workspace features 12 dedicated analytical modules with custom dark and light themes. In the sidebar, the user can select their forecasting model, set the projection horizon in terms of future observed records, configure scenario capacity, and tune warning thresholds. [BREATH]

On the **Overview** page, we see immediate executive KPIs: the latest observed load of 2,484 children, the historical average of 3,554, and the 7-step forecast trajectory. 

Switching to the **Forecast Explorer**, we can inspect the forecast alongside empirical 90% error bands, with interactive zoom and tooltips directly illustrating where future demand tracks relative to warning thresholds. [PAUSE]

In the **Model Evaluation** page, stakeholders can toggle between walk-forward cross-validation metrics and chronological holdout splits. The **Backtesting Simulator** allows analysts to replay historical prediction origins step-by-step, auditing where individual model residuals spiked during past inflection points. [BREATH]

Next, the **Capacity Planner** evaluates shelter demand against capacity. With our current 2,484 load and a 5,000 bed threshold, current utilization is roughly 50%, designated as 'Low Risk'. 

In the **Scenario Lab**, we test hypothetical stress multipliers — a 10% elevated demand shock and a 25% severe surge scenario. Even under a 25% shock, projected peak demand reaches 3,125 children, remaining safely below the 4,000-bed warning line. [PAUSE]

Finally, the **Data Quality** page transparently communicates date coverage and gap distributions, while the **Report Center** enables single-click export of executive PDF briefs generated dynamically with ReportLab and multi-sheet Excel workbooks."

---

### [PART 7 — EMPIRICAL UNCERTAINTY & OPERATIONAL CAPACITY | 5:35 – 6:20]
*(Tone: Rigorous, responsible, transparent)*

"A critical pillar of this project is responsible uncertainty communication. [PAUSE]

Rather than assuming Gaussian residuals that rarely hold in real operations, we derived empirical error quantiles — 80%, 90%, and 95% — from 315 out-of-sample backtest errors. On evaluation, the uncalibrated 90% interval achieved 86% empirical coverage, while our chronological calibration phase safely encompassed 100% of evaluation points. [BREATH]

I must also emphasize an essential operational disclaimer: the 5,000 capacity threshold used throughout the dashboard is an illustrative planning benchmark, not an officially certified facility ceiling. In a live deployment, this input is dynamically updated with facility-specific operational bed counts."

---

### [PART 8 — CONCLUSION & KEY TAKEAWAYS | 6:20 – 6:50]
*(Tone: Inspiring, professional, conclusive)*

"To summarize, Care Load Intelligence provides an end-to-end framework combining rigorous data cleaning, multi-horizon time-series evaluation, empirical uncertainty quantification, and an interactive decision-support application. [PAUSE]

All source code, automated test suites, precomputed reports, and educational Jupyter notebooks are publicly available on my GitHub repository, and you can test the live application at the link below. 

Thank you very much for your time, and I welcome your questions and feedback!"

---

## 2. Executive Voiceover Script (2-Minute Demo)

*(Pacing: 135 words per minute | Duration: ~120 seconds)*

"Hi, I’m Utkarsh Sengar. Welcome to a two-minute walkthrough of **Care Load Intelligence**, an applied time-series forecasting and capacity planning platform.

In custodial care and humanitarian placement programs, shelter managers face volatile demand patterns and irregular operational reporting. Using historical Department of Health and Human Services data spanning from January 2023 to December 2025, I developed an end-to-end analytics platform to forecast placement demand and stress-test shelter capacity.

The raw data contains 720 observed records across 1,075 calendar days. Because reporting pauses on weekends and holidays, missing dates cannot simply be imputed with zero. Our pipeline treats the series as an irregular sequence of observed records and parallel weekly aggregations.

We evaluated Naive persistence, ARIMA(1,1,1), and Holt's Exponential Smoothing models. In our 45-origin rolling walk-forward evaluation, **Holt's Damped Trend model delivered the best short-horizon accuracy**, achieving a Mean Absolute Error of 78.9 children — outperforming both ARIMA and Naive baselines. Damping the trend prevented explosive over-projections while adapting to local shifts.

On the live Streamlit dashboard, decision-makers can select forecasting models, adjust horizons from 3 to 30 records, and inspect empirical 80%, 90%, and 95% uncertainty intervals computed from historical residual quantiles.

The dashboard also features an interactive **Scenario Lab**, enabling planners to test 10% and 25% demand surge assumptions against configurable capacity thresholds, with one-click export of executive PDF and Excel reports.

The complete codebase, reproducible Jupyter notebooks, and live application are available on my GitHub. Thank you!"

---

## 3. High-Impact Portfolio Voiceover (60 Seconds)

*(Pacing: 140 words per minute | Duration: ~60 seconds)*

"Can machine learning help humanitarian care facilities anticipate demand spikes before capacity bottlenecks occur?

I built **Care Load Intelligence** — an end-to-end forecasting platform analyzing 720 historical operational records from 2023 through 2025.

Instead of naïve imputation that misinterprets irregular weekend gaps as zero caseload, I engineered a sequence-aware forecasting pipeline.

In a 45-origin rolling walk-forward backtest, **Holt's Damped Trend achieved top performance with an MAE of 78.9 children**, successfully capturing momentum without the runaway trend errors of linear models.

The deployed Streamlit web application features 12 analytical pages, empirical uncertainty intervals, demand stress-testing for 10% and 25% surge scenarios, and automated PDF report generation.

Explore the live interactive dashboard and GitHub repository in the links below. I’m Utkarsh Sengar — thanks for watching!"
