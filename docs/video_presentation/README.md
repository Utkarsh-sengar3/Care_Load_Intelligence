# Video Presentation Production Guide

This directory contains complete scripts, shot lists, audio guidance, and visual assets for producing a professional video demonstration of **Care Load Intelligence**.

---

## Directory Contents

- **`VIDEO_SCRIPT.md`**: Master video script containing 3 versions:
  - 5–7 Minute Master Technical Presentation
  - 2-Minute Concise Executive Demo
  - 60-Second High-Impact Portfolio Pitch
- **`VOICEOVER_SCRIPT.md`**: Teleprompter/recording script with phonetic pronunciation guide and performance cues.
- **`SCREEN_RECORDING_SHOTLIST.md`**: Actionable step-by-step recording sequence detailing exact mouse movements, clicks, and zoom targets.
- **`assets/`**: High-resolution (300 DPI) charts and architectural diagrams generated directly from real project data and models:
  - `01_historical_care_load_trend.png`
  - `02_model_performance_comparison.png`
  - `03_forecast_uncertainty_intervals.png`
  - `04_capacity_stress_scenarios.png`
  - `05_horizon_error_sensitivity.png`
  - `06_ml_workflow_architecture.png`
  - `07_actual_vs_predicted_scatter.png`
  - `08_application_architecture_diagram.png`

---

## Technical Recording Specifications

| Setting | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Resolution** | 1920 × 1080 (1080p) | 16:9 widescreen standard |
| **Frame Rate** | 60 fps (or 30 fps) | 60 fps provides ultra-smooth cursor motion |
| **Audio Format** | 48 kHz / 24-bit WAV | Crisp vocal clarity without clipping |
| **Bitrate** | 12,000–16,000 kbps (CBR/VBR) | Crisp text rendering on charts and tables |
| **Audio Filter Chain** | High-pass at 80Hz, Noise Gate (-45dB), Compressor (3:1 ratio), Limiter (-1dB) | Broadcast vocal standard |

---

## Recommended Production Workflow

### Step 1: Voiceover Audio Recording
1. Open `VOICEOVER_SCRIPT.md`.
2. Use Audacity, OBS Studio, or your preferred DAW.
3. Record in a quiet, treated space. Maintain consistent 4–6 inches distance from the microphone with a pop filter.
4. Record 1–2 complete takes. Use the pronunciation guide for technical acronyms (ARIMA, HHS, CBP, MAE, MAPE).
5. Clean background noise, apply mild vocal normalization to -14 LUFS, and export `master_voiceover.wav`.

### Step 2: Screen Recording (Live App & Notebook)
1. Open `SCREEN_RECORDING_SHOTLIST.md`.
2. Launch OBS Studio or Camtasia configured to record your primary 1080p display.
3. Open the live website: `https://cli-utkaresh-sengar-updated.streamlit.app/`.
4. Maximize the browser, set zoom to 100% or 110%, and select the Dark theme.
5. Perform the exact mouse clicks, tab navigations, and download actions specified in the shot list while listening to your recorded voiceover track.
6. Record separate B-roll clips of the Jupyter Notebook (`Care_Load_Intelligence_End_to_End_ML.ipynb`) executing cells.

### Step 3: Video Assembly & Editing
1. In DaVinci Resolve, Premiere Pro, or CapCut:
   - Place `master_voiceover.wav` on Audio Track 1.
   - Align screen capture clips on Video Track 1 following the timestamps in `VIDEO_SCRIPT.md`.
   - Insert PNG chart assets from `assets/` on Video Track 2 during technical explanations. Use subtle 1.05x slow zooms (Ken Burns effect).
   - Add lower-third callouts for presenter name (*Utkarsh Sengar*), key metrics (*MAE: 78.9*), and operational disclaimers (*Illustrative 5,000 Bed Capacity*).
2. Export the final video as `H.264 / MP4` at 1080p.

---

## Ready-to-Use Video Metadata

### YouTube Title Options:
1. `Care Load Intelligence: Predictive Time-Series Forecasting & Placement Demand Analysis`
2. `End-to-End Time-Series Machine Learning in Production: Care Load Intelligence Walkthrough`
3. `Forecasting Humanitarian Care Load: From Irregular Data to Interactive Streamlit App`

### YouTube Description:
```markdown
Care Load Intelligence is an applied time-series forecasting and decision-support platform built on historical HHS Unaccompanied Alien Children operational data (2023–2025).

In this video, I walk through the complete machine learning architecture:
- Auditing 720 irregular operational records across 1,075 calendar days
- Why missing reporting dates should NEVER be imputed as zero
- Comparing Naive benchmarks, ARIMA(1,1,1), and Holt's Damped Trend models
- Resolving the evaluation paradox: Walk-Forward Validation (MAE 78.9) vs. Chronological Holdout
- Quantifying empirical uncertainty with 80%, 90%, and 95% historical error quantiles
- Interactive live demonstration of the Streamlit decision-support application, featuring capacity stress-testing and automated PDF/Excel reporting

🔗 Live Application: https://cli-utkaresh-sengar-updated.streamlit.app/
💻 GitHub Repository: https://github.com/Utkarsh-sengar3/Care_Load_Intelligence_Updated_Website
👤 Presenter: Utkarsh Sengar

#MachineLearning #TimeSeries #Forecasting #Streamlit #DataScience #Python #AI
```

### LinkedIn Post Copy:
```markdown
Excited to share my latest applied machine learning project: Care Load Intelligence! 🚀

Managing humanitarian and transitional care systems requires understanding historical population dynamics, anticipating demand surges, and systematically evaluating facility capacity.

Key highlights of this project:
📊 Data Engineering: Modeled 720 irregular operational records (2023–2025) without artificial zero-imputation.
🧠 Model Architecture: Benchmarked Naive persistence, ARIMA(1,1,1), and Holt's Damped Trend.
🎯 Key Finding: Holt's Damped Trend won the 45-origin rolling walk-forward evaluation (MAE: 78.9), while simple persistence outperformed over long single-origin holdouts—demonstrating how evaluation design dictates model selection.
🛡️ Uncertainty Quantification: Implemented empirical error quantiles (80%, 90%, 95%) calibrated across historical backtest residuals.
🌐 Production Dashboard: Deployed a 12-page Streamlit application with demand stress-testing (+10%, +25%) and dynamic ReportLab PDF export.

Check out the live interactive app and open-source GitHub repository below!

Live App: https://cli-utkaresh-sengar-updated.streamlit.app/
GitHub: https://github.com/Utkarsh-sengar3/Care_Load_Intelligence_Updated_Website

Feedback and discussions are welcome!
```
