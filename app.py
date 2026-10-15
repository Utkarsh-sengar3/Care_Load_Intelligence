from pathlib import Path
from io import BytesIO
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from forecasting_engine import load_care_data, run_forecasting_pipeline, generate_forecast

REPORTS = ROOT / "reports"
TARGET = "Children in HHS Care"

st.set_page_config(page_title="Care Load Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#e8eef8;--muted:#94a3b8;--line:rgba(148,163,184,.17);--panel:#101a2b;--canvas:#0a1220;--blue:#64a8ff;--cyan:#5dd6d0;--amber:#f3c77b}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 75% 0%,rgba(40,91,143,.14),transparent 34%),var(--canvas);color:var(--ink)}
[data-testid="stHeader"]{background:rgba(10,18,32,.85)}
[data-testid="stSidebar"]{background:#0c1626;border-right:1px solid var(--line)}
.block-container{max-width:1500px;padding:1.8rem 2.3rem 3rem}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:#f3f7ff}
.brand{display:flex;gap:11px;align-items:center;padding:5px 4px 19px;border-bottom:1px solid var(--line);margin-bottom:20px}
.brand-mark{width:39px;height:39px;border-radius:12px;display:grid;place-items:center;color:#071322;font-weight:800;font-size:22px;background:linear-gradient(135deg,var(--cyan),var(--blue))}
.brand-title{color:#f3f7ff;font-family:'Manrope',sans-serif;font-weight:800;font-size:15px;line-height:1.25}
.brand-subtitle,.muted{color:var(--muted);font-size:11px;margin-top:4px;letter-spacing:.08em;text-transform:uppercase}
.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.13em;font-size:.72rem;font-weight:700;margin-bottom:8px}
.subtitle{color:var(--muted);font-size:.95rem;margin-top:-7px;margin-bottom:24px;line-height:1.6}
.hero{background:linear-gradient(115deg,rgba(24,43,68,.96),rgba(15,30,49,.92));border:1px solid rgba(100,168,255,.23);border-radius:20px;padding:27px 30px;margin:4px 0 22px}
.hero-title{font-family:'Manrope',sans-serif;font-weight:800;font-size:clamp(1.6rem,3vw,2.4rem);letter-spacing:-.045em;color:#f5f8ff;margin:0 0 8px;line-height:1.15}
.hero-copy{color:#b2c2d7;max-width:730px;line-height:1.65;margin-bottom:15px}
.pill{display:inline-block;border:1px solid rgba(93,214,208,.27);background:rgba(93,214,208,.08);color:#9be9e3;padding:5px 10px;border-radius:999px;font-size:.75rem;font-weight:600}
.section-title{font-family:'Manrope',sans-serif;font-weight:750;color:#edf4ff;font-size:1.12rem;margin:24px 0 5px}
.section-note{color:var(--muted);font-size:.8rem;margin-bottom:12px}
[data-testid="stMetric"]{background:linear-gradient(145deg,#121f32,#0f1a2a);border:1px solid var(--line);border-radius:15px;padding:18px 19px;min-height:116px}
[data-testid="stMetricLabel"]{color:#9dafc5!important;font-size:.82rem!important}
[data-testid="stMetricValue"]{color:#f1f6ff!important;font-family:'Manrope',sans-serif;font-weight:750;font-size:1.65rem!important}
.notice{background:rgba(243,199,123,.07);border:1px solid rgba(243,199,123,.22);border-radius:12px;padding:13px 15px;color:#dfd0b4;font-size:.85rem;line-height:1.55;margin:12px 0}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;border:1px solid rgba(100,168,255,.35);font-weight:600}
div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
hr{border-color:var(--line)} footer{visibility:hidden}
@media(max-width:800px){.block-container{padding:1rem 1rem 2rem}.hero{padding:21px}}
</style>
""", unsafe_allow_html=True)

def layout(fig, height=360):
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=12,r=12,t=45,b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",color="#b9c8db",size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=1.02),
        xaxis=dict(showgrid=False,zeroline=False,linecolor="rgba(148,163,184,.15)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.12)",zeroline=False),
        hoverlabel=dict(bgcolor="#101a2b",bordercolor="#35516e",font_color="#f3f7ff")
    )
    return fig

@st.cache_data(show_spinner=False)
def history_data():
    return load_care_data()

@st.cache_data(show_spinner=False)
def csv_data(path):
    p=Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def forecast_data(model,horizon,capacity,warning):
    return run_forecasting_pipeline(model_name=model,horizon=int(horizon),capacity=int(capacity),warning_utilization=float(warning))


def error_metrics(actual, predicted):
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float)
    err=actual-predicted
    nonzero=np.abs(actual)>1e-9
    return {"MAE":float(np.mean(np.abs(err))),"RMSE":float(np.sqrt(np.mean(err**2))),"MAPE (%)":float(np.mean(np.abs(err[nonzero]/actual[nonzero]))*100) if nonzero.any() else np.nan,"Mean error (actual − predicted)":float(np.mean(err))}

def excel_report(sheets):
    buffer=BytesIO()
    with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
        for name,frame in sheets.items():
            frame.to_excel(writer,sheet_name=name[:31],index=False)
    buffer.seek(0)
    return buffer.getvalue()

def pdf_report(summary_frame, forecast_frame):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=landscape(letter),rightMargin=0.45*inch,leftMargin=0.45*inch,topMargin=0.45*inch,bottomMargin=0.45*inch)
    styles=getSampleStyleSheet(); story=[]
    story.append(Paragraph("Care Load Intelligence — Forecast Report",styles["Title"]))
    story.append(Paragraph("Analytical prototype · Generated locally from the selected project data",styles["Normal"]))
    story.append(Spacer(1,12))
    summary=summary_frame.iloc[0]
    summary_rows=[["Report item","Value"],["Generated at",str(summary["generated_at"])],["Selected model",str(summary["model"])],["Forecast horizon",f"{summary['forecast_horizon_future_observed_records']} future observed records"],["Forecast origin",str(summary["forecast_origin"])],["Latest observed load",f"{summary['latest_observed_load']:,.0f}"],["Mean forecast",f"{summary['mean_forecast']:,.1f}"],["Peak forecast",f"{summary['peak_forecast']:,.1f}"],["Planning capacity assumption",f"{summary['planning_capacity_assumption']:,.0f} (not verified official capacity)"],["Peak utilization",f"{summary['peak_utilization_percent']:.1f}%"]]
    t=Table(summary_rows,colWidths=[2.4*inch,6.8*inch],repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#d5dde7")),("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f5f7fa")),("PADDING",(0,0),(-1,-1),6)]))
    story.append(t); story.append(Spacer(1,14)); story.append(Paragraph("Forecast values",styles["Heading2"]))
    cols=[c for c in ["Forecast_Step","Forecast_Care_Load","Lower_Q80","Upper_Q80","Lower_Q90","Upper_Q90","Lower_Q95","Upper_Q95","Capacity_Utilization_Percent","Scenario_Risk"] if c in forecast_frame.columns]
    data=[cols]+forecast_frame[cols].round(2).astype(str).values.tolist()
    ft=Table(data,repeatRows=1)
    ft.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#18324d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#d5dde7")),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),4)]))
    story.append(ft); story.append(Spacer(1,12))
    story.append(Paragraph("Limitations: observation dates are irregular; forecast steps mean future observed records, not calendar days. Empirical error ranges are not guaranteed prediction intervals. Demand scenarios are hypothetical, and the capacity input is not verified official capacity. Do not use this prototype as the sole basis for operational decisions.",styles["BodyText"]))
    doc.build(story); buffer.seek(0); return buffer.getvalue()

with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-title">Care Load<br>Intelligence</div><div class="brand-subtitle">Forecasting workspace</div></div></div>""",unsafe_allow_html=True)
    page=st.radio("WORKSPACE",["Overview","Forecast explorer","Model evaluation","Backtesting simulator","Uncertainty","Capacity planner","Scenario lab","Horizon sensitivity","Data quality","Report center","Data explorer","Methodology"])
    st.markdown("---")
    theme=st.selectbox("Appearance",["Dark · analyst workspace","Light · report view"],index=0)
    if theme.startswith("Light"):
        st.markdown("<style>.stApp{background:#f5f7fa!important;color:#172033!important}[data-testid=stSidebar]{background:#edf1f6!important}h1,h2,h3,.section-title{color:#172033!important}[data-testid=stMetric]{background:#fff!important;border-color:#dce3ec!important}[data-testid=stMetricValue]{color:#172033!important}[data-testid=stMetricLabel]{color:#526176!important}.hero{background:#e8eef6!important;border-color:#d2deed!important}.hero-title{color:#172033!important}.hero-copy{color:#526176!important}.section-note,.subtitle{color:#526176!important}</style>",unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Forecast controls</div>',unsafe_allow_html=True)
    model=st.selectbox("Forecast model",["Holt_Damped","Naive","ARIMA_1_1_1"],index=0)
    horizon=st.select_slider("Future observed records",options=[3,7,14,30],value=7)
    capacity=st.number_input("Scenario capacity",min_value=1,max_value=1000000,value=5000,step=250)
    warning_pct=st.slider("Warning threshold",50,95,80,5)
    warning=warning_pct/100
    st.markdown("---")
    st.caption("Capacity is a configurable scenario, not a verified official limit.")

try:
    hist=history_data()
    fc,scenarios,summary=forecast_data(model,horizon,capacity,warning)
except Exception as e:
    st.error(f"Could not load forecasting workspace: {e}")
    st.stop()

last_date=pd.to_datetime(hist["Date"]).max()
latest=float(hist[TARGET].iloc[-1])
mean_load=float(hist[TARGET].mean())
max_load=float(hist[TARGET].max())
peak=float(fc["Forecast_Care_Load"].max())

walk=csv_data(str(REPORTS/"phase3_4_advanced_model_comparison.csv"))
holdout=csv_data(str(REPORTS/"phase4_1_final_model_comparison.csv"))
predictions=csv_data(str(REPORTS/"phase3_4_advanced_predictions.csv"))

st.markdown('<div class="eyebrow">UAC · Predictive analytics</div>',unsafe_allow_html=True)

if page=="Overview":
    st.markdown(f"""<div class="hero"><div class="hero-title">Care demand, viewed with context.</div><div class="hero-copy">Explore observed patterns, compare forecasting approaches, and test capacity assumptions in one analytical workspace.</div><span class="pill">Historical data loaded</span><span style="color:#8fa7c3;font-size:.78rem;margin-left:10px">Latest observation · {last_date:%d %b %Y}</span></div>""",unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Latest observed load",f"{latest:,.0f}")
    b.metric("Historical mean",f"{mean_load:,.0f}")
    c.metric("Historical maximum",f"{max_load:,.0f}")
    d.metric(f"Mean forecast · {horizon} records",f"{fc['Forecast_Care_Load'].mean():,.0f}")
    st.markdown('<div class="section-title">Observed care load</div><div class="section-note">All available records · observation dates are irregular</div>',unsafe_allow_html=True)
    fig=px.line(hist,x="Date",y=TARGET,labels={"Date":"Observation date",TARGET:"Children in HHS Care"})
    fig.update_traces(line=dict(color="#64a8ff",width=2),hovertemplate="%{x|%d %b %Y}<br>Load: %{y:,.0f}<extra></extra>")
    st.plotly_chart(layout(fig,390),use_container_width=True)
    left,right=st.columns([1.35,1])
    with left:
        st.markdown('<div class="section-title">Forecast trajectory</div><div class="section-note">Selected model · future observed records</div>',unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Forecast_Care_Load"],mode="lines+markers",name="Point forecast",line=dict(color="#5dd6d0",width=3),marker=dict(size=7)))
        if fc["Lower_Q90"].notna().any():
            fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Upper_Q90"],mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Lower_Q90"],mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(93,214,208,.12)",name="Empirical Q90 error range",hoverinfo="skip"))
        fig.add_hline(y=capacity,line_dash="dash",line_color="#f3c77b",annotation_text="Scenario capacity")
        fig.update_xaxes(dtick=1,title="Future observed record")
        st.plotly_chart(layout(fig,330),use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Capacity snapshot</div><div class="section-note">Based on the selected threshold</div>',unsafe_allow_html=True)
        st.metric("Scenario capacity",f"{capacity:,.0f}")
        st.metric("Peak forecast utilization",f"{peak/capacity*100:.1f}%")
        st.metric("Estimated extra places at peak",f"{int(fc['Estimated_Additional_Places'].max()):,}")
        st.markdown('<div class="notice">Capacity is an editable scenario input. This result does not establish actual HHS capacity or operational risk.</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Workspace guide</div>',unsafe_allow_html=True)
    x,y,z=st.columns(3)
    x.markdown("**Forecast explorer**"); x.caption("Inspect predictions and export a forecast table.")
    y.markdown("**Model evaluation**"); y.caption("Review walk-forward and holdout results separately.")
    z.markdown("**Scenario lab**"); z.caption("Compare hypothetical demand assumptions.")

elif page=="Forecast explorer":
    st.title("Forecast explorer")
    st.markdown('<div class="subtitle">Inspect point forecasts, thresholds, and empirical error ranges.</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Selected model",model.replace("_"," "))
    b.metric("Forecast horizon",f"{horizon} records")
    c.metric("Forecast range",f"{fc['Forecast_Care_Load'].min():,.0f}–{fc['Forecast_Care_Load'].max():,.0f}")
    st.markdown('<div class="section-title">Observed history + future forecast</div><div class="section-note">The dashed boundary marks the forecast origin. Future steps are not calendar dates.</div>',unsafe_allow_html=True)
    fig=go.Figure()
    recent=hist.tail(min(90,len(hist)))
    fig.add_trace(go.Scatter(x=recent["Date"],y=recent[TARGET],mode="lines",name="Observed care load",line=dict(color="#64a8ff",width=2)))
    fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Forecast_Care_Load"],mode="lines+markers",name="Forecast · observed-record steps",line=dict(color="#5dd6d0",width=3)))
    fig.add_hline(y=capacity,line_dash="dash",line_color="#f3c77b",annotation_text="Planning capacity")
    st.plotly_chart(layout(fig,380),use_container_width=True)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Forecast_Care_Load"],mode="lines+markers",name="Point forecast",line=dict(color="#64a8ff",width=3)))
    if fc["Lower_Q90"].notna().any():
        fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Upper_Q90"],mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Lower_Q90"],mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(100,168,255,.14)",name="Empirical Q90 error range"))
    fig.add_hline(y=capacity,line_dash="dash",line_color="#f3c77b",annotation_text="Capacity scenario")
    fig.add_hline(y=capacity*warning,line_dash="dot",line_color="#5dd6d0",annotation_text="Warning threshold")
    fig.update_xaxes(dtick=1,title="Future observed record")
    st.plotly_chart(layout(fig,420),use_container_width=True)
    cols=[c for c in ["Forecast_Step","Forecast_Care_Load","Lower_Q80","Upper_Q80","Lower_Q90","Upper_Q90","Lower_Q95","Upper_Q95","Capacity_Utilization_Percent","Estimated_Additional_Places","Scenario_Risk"] if c in fc.columns]
    st.dataframe(fc[cols].round(2),use_container_width=True,hide_index=True)
    st.download_button("Download forecast CSV",fc.to_csv(index=False).encode(),file_name="care_load_forecast.csv",mime="text/csv",type="primary")
    st.caption(f"Forecast origin: {last_date:%d %b %Y}. Future calendar dates are not inferred.")

elif page=="Model evaluation":
    st.title("Model evaluation")
    st.markdown('<div class="subtitle">Validation designs answer different questions. Review them separately.</div>',unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["Walk-forward","Final holdout","Prediction errors"])
    with tab1:
        if walk.empty: st.info("Run Phase 3.4 to create the model comparison report.")
        else:
            st.dataframe(walk,use_container_width=True,hide_index=True)
            if {"Model","MAE"}.issubset(walk.columns):
                fig=px.bar(walk,x="Model",y="MAE",labels={"MAE":"Mean absolute error"})
                fig.update_traces(marker_color="#64a8ff")
                st.plotly_chart(layout(fig,340),use_container_width=True)
            st.caption("Repeated seven-observation forecasts with expanding training windows.")
    with tab2:
        if holdout.empty: st.info("Run Phase 4.1 to create the final holdout report.")
        else:
            st.dataframe(holdout,use_container_width=True,hide_index=True)
            metric=next((x for x in ["MAE","RMSE","MAPE"] if x in holdout.columns),None)
            if metric and "Model" in holdout.columns:
                fig=px.bar(holdout,x="Model",y=metric,title=f"Final holdout · {metric}")
                fig.update_traces(marker_color="#5dd6d0")
                st.plotly_chart(layout(fig,340),use_container_width=True)
            st.caption("Single-origin multi-step evaluation on the chronological test segment.")
    with tab3:
        if predictions.empty or not {"Model","Actual","Prediction"}.issubset(predictions.columns):
            st.info("Historical prediction file is unavailable or has different columns.")
        else:
            err=predictions.copy()
            err["Error (actual − predicted)"]=err["Actual"]-err["Prediction"]
            names=sorted(err["Model"].dropna().unique().tolist())
            chosen=st.selectbox("Inspect model errors",names)
            sub=err[err["Model"]==chosen]
            fig=px.scatter(sub,x="Prediction",y="Error (actual − predicted)",hover_data=[x for x in ["Forecast_Step","Date"] if x in sub.columns])
            fig.add_hline(y=0,line_dash="dash",line_color="#f3c77b")
            st.plotly_chart(layout(fig,350),use_container_width=True)
            st.dataframe(sub.head(250),use_container_width=True,hide_index=True)
    st.markdown('<div class="notice">A model that wins on short-horizon walk-forward evaluation may not win on a long-horizon holdout.</div>',unsafe_allow_html=True)

elif page=="Backtesting simulator":
    st.title("Backtesting simulator")
    st.markdown('<div class="subtitle">Replay a forecast from a historical cutoff and compare it only with observations that came afterwards.</div>',unsafe_allow_html=True)
    max_train=max(100,len(hist)-31)
    default_train=min(max_train,max(100,int(len(hist)*0.8)))
    train_count=st.slider("Training observations before forecast origin",min_value=100,max_value=max_train,value=default_train,step=10)
    max_h=min(30,len(hist)-train_count)
    test_h=st.slider("Evaluation horizon · future observed records",min_value=1,max_value=max_h,value=min(7,max_h))
    bt_model=st.selectbox("Backtest model",["Holt_Damped","Naive","ARIMA_1_1_1"],index=["Holt_Damped","Naive","ARIMA_1_1_1"].index(model),key="bt_model")
    if st.button("Run historical backtest",type="primary"):
        with st.spinner("Fitting model on the selected historical window…"):
            train=hist.iloc[:train_count].copy(); actual=hist.iloc[train_count:train_count+test_h].copy()
            pred=generate_forecast(train,model_name=bt_model,horizon=test_h)
            pred["Actual"] = actual[TARGET].to_numpy()
            pred["Observation_Date"] = actual["Date"].to_numpy()
            pred["Error (actual − predicted)"]=pred["Actual"]-pred["Forecast_Care_Load"]
            pred["Absolute_Error"]=pred["Error (actual − predicted)"].abs()
            metrics=error_metrics(pred["Actual"],pred["Forecast_Care_Load"])
            st.session_state["care_backtest"]={"pred":pred,"metrics":metrics,"model":bt_model,"origin":train["Date"].iloc[-1],"end":actual["Date"].iloc[-1]}
    result=st.session_state.get("care_backtest")
    if result:
        pred=result["pred"]; metrics=result["metrics"]
        st.caption(f"Model: {result['model']} · Forecast origin: {pd.Timestamp(result['origin']):%d %b %Y} · Last evaluated observation: {pd.Timestamp(result['end']):%d %b %Y}")
        a,b,c,d=st.columns(4); a.metric("MAE",f"{metrics['MAE']:,.2f}"); b.metric("RMSE",f"{metrics['RMSE']:,.2f}"); c.metric("MAPE",f"{metrics['MAPE (%)']:.2f}%" if pd.notna(metrics['MAPE (%)']) else "—"); d.metric("Mean error",f"{metrics['Mean error (actual − predicted)']:,.2f}")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=pred["Observation_Date"],y=pred["Actual"],mode="lines+markers",name="Actual observations",line=dict(color="#64a8ff",width=2))); fig.add_trace(go.Scatter(x=pred["Observation_Date"],y=pred["Forecast_Care_Load"],mode="lines+markers",name="Backtest forecast",line=dict(color="#5dd6d0",width=2,dash="dash"))); st.plotly_chart(layout(fig,390),use_container_width=True)
        st.dataframe(pred,use_container_width=True,hide_index=True); st.download_button("Download backtest CSV",pred.to_csv(index=False).encode(),file_name="historical_backtest.csv",mime="text/csv")
        st.info("This is a single-origin historical replay, not a replacement for the project's separate walk-forward and final-holdout evaluations.")
    else: st.caption("Choose a cutoff and run the backtest to create a reproducible historical comparison.")

elif page=="Horizon sensitivity":
    st.title("Forecast horizon sensitivity")
    st.markdown('<div class="subtitle">Compare how the selected model’s forecast changes as the number of future observed records increases.</div>',unsafe_allow_html=True)
    horizons=st.multiselect("Horizons to compare",[3,7,14,30],default=[3,7,14,30])
    if not horizons: st.info("Select at least one horizon.")
    else:
        with st.spinner("Generating forecasts for selected horizons…"):
            frames=[]
            for h in horizons:
                f,_,_=forecast_data(model,h,capacity,warning); f["Selected_Horizon"]=h; frames.append(f)
            combined=pd.concat(frames,ignore_index=True)
        fig=go.Figure()
        for h in horizons:
            part=combined[combined["Selected_Horizon"]==h]
            fig.add_trace(go.Scatter(x=part["Forecast_Step"],y=part["Forecast_Care_Load"],mode="lines+markers",name=f"Horizon {h}"))
        fig.add_hline(y=capacity,line_dash="dash",line_color="#f3c77b",annotation_text="Planning capacity")
        fig.update_xaxes(title="Future observed record",dtick=1); st.plotly_chart(layout(fig,400),use_container_width=True)
        st.dataframe(combined,use_container_width=True,hide_index=True); st.download_button("Download horizon comparison",combined.to_csv(index=False).encode(),file_name="horizon_sensitivity.csv",mime="text/csv")
        st.caption("Horizon steps refer to future observed records, not calendar days. Different horizons are model outputs, not validated accuracy comparisons.")

elif page=="Data quality":
    st.title("Data quality monitor")
    st.markdown('<div class="subtitle">Check data completeness and reporting cadence before interpreting forecasts.</div>',unsafe_allow_html=True)
    raw_path=ROOT/"data"/"processed"/"uac_observed_forecasting_data.csv"
    raw=pd.read_csv(raw_path) if raw_path.exists() else hist.copy()
    raw_dates=pd.to_datetime(raw.get("Date",pd.Series(dtype="object")),errors="coerce")
    raw_target=pd.to_numeric(raw.get(TARGET,pd.Series(dtype="object")),errors="coerce")
    sorted_dates=raw_dates.dropna().sort_values()
    first=sorted_dates.min() if len(sorted_dates) else pd.NaT; last=sorted_dates.max() if len(sorted_dates) else pd.NaT
    span=(last-first).days+1 if pd.notna(first) and pd.notna(last) else len(raw)
    unique_dates=sorted_dates.nunique(); missing_calendar=max(0,span-unique_dates)
    dup_dates=int(raw_dates.duplicated().sum()); invalid_dates=int(raw_dates.isna().sum()); missing_target=int(raw_target.isna().sum())
    a,b,c,d=st.columns(4); a.metric("Loaded observations",f"{len(hist):,}"); b.metric("Missing calendar dates",f"{missing_calendar:,}"); c.metric("Duplicate date rows",f"{dup_dates:,}"); d.metric("Missing target values",f"{missing_target:,}")
    gaps=sorted_dates.sort_values().diff().dt.days.dropna()
    left,right=st.columns([1.2,1])
    with left:
        if len(gaps):
            gap_df=pd.DataFrame({"Gap in calendar days":gaps.astype(int)})
            fig=px.histogram(gap_df,x="Gap in calendar days",nbins=25,title="Intervals between recorded dates")
            st.plotly_chart(layout(fig,330),use_container_width=True)
    with right:
        st.markdown("**Audit findings**")
        st.write(f"Date range: **{first:%d %b %Y} – {last:%d %b %Y}**" if pd.notna(first) and pd.notna(last) else "Date range unavailable.")
        st.write(f"Invalid date values in raw file: **{invalid_dates}**")
        st.write(f"Missing target values in raw file: **{missing_target}**")
        st.write(f"Median gap between dated rows: **{gaps.median():.0f} calendar days**" if len(gaps) else "Gap statistics unavailable.")
    if "Date" in raw.columns and len(sorted_dates):
        cadence=pd.DataFrame({"Date":sorted_dates.dt.day_name().value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],fill_value=0).index,"Observation count":sorted_dates.dt.day_name().value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],fill_value=0).values})
        fig=px.bar(cadence,x="Date",y="Observation count",title="Reporting weekday distribution")
        st.plotly_chart(layout(fig,300),use_container_width=True)
    st.warning("A missing calendar date does not mean zero care load. This dataset has irregular reporting dates; forecasts are indexed by future observed records, not calendar days.")
    st.download_button("Download raw data audit",pd.DataFrame([{"raw_rows":len(raw),"loaded_valid_observations":len(hist),"missing_calendar_dates":missing_calendar,"duplicate_date_rows":dup_dates,"invalid_dates":invalid_dates,"missing_target_values":missing_target,"first_date":str(first),"last_date":str(last)}]).to_csv(index=False).encode(),file_name="data_quality_audit.csv",mime="text/csv")

elif page=="Report center":
    st.title("Report center")
    st.markdown('<div class="subtitle">Export a reproducible snapshot of the selected forecast, assumptions and available evaluation reports.</div>',unsafe_allow_html=True)
    report_summary=pd.DataFrame([{"generated_at":datetime.now().isoformat(timespec="seconds"),"model":model,"forecast_horizon_future_observed_records":horizon,"forecast_origin":str(last_date.date()),"latest_observed_load":latest,"mean_forecast":float(fc["Forecast_Care_Load"].mean()),"peak_forecast":peak,"planning_capacity_assumption":capacity,"warning_threshold_percent":warning_pct,"peak_utilization_percent":peak/capacity*100,"capacity_is_verified":False}])
    st.subheader("Report preview")
    st.dataframe(report_summary,use_container_width=True,hide_index=True)
    st.caption("The report includes assumptions and limitations; capacity is explicitly recorded as an unverified planning input.")
    xlsx=excel_report({"Executive Summary":report_summary,"Forecast":fc,"Scenarios":scenarios,"Walk Forward":walk if not walk.empty else pd.DataFrame(),"Final Holdout":holdout if not holdout.empty else pd.DataFrame(),"Historical Data":hist})
    st.download_button("Download Excel report (.xlsx)",xlsx,file_name="care_load_forecasting_report.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary")
    pdf_bytes=pdf_report(report_summary,fc)
    st.download_button("Download PDF report",pdf_bytes,file_name="care_load_forecasting_report.pdf",mime="application/pdf")
    st.download_button("Download executive summary CSV",report_summary.to_csv(index=False).encode(),file_name="care_load_executive_summary.csv",mime="text/csv")
    st.download_button("Download forecast CSV",fc.to_csv(index=False).encode(),file_name="care_load_forecast.csv",mime="text/csv")
    st.markdown("**Included workbook sheets:** Executive Summary, Forecast, Scenarios, Walk Forward, Final Holdout, Historical Data.")
    st.info("This report snapshot is generated locally from the current project data and reports. It is not an official operational report.")

elif page=="Uncertainty":
    st.title("Forecast uncertainty")
    st.markdown('<div class="subtitle">Historical out-of-sample errors help describe forecast variability.</div>',unsafe_allow_html=True)
    valid=fc.dropna(subset=["Lower_Q90","Upper_Q90"]).copy()
    if valid.empty: st.warning("No empirical error ranges are available for this model and horizon.")
    else:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=valid["Forecast_Step"],y=valid["Upper_Q90"],mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=valid["Forecast_Step"],y=valid["Lower_Q90"],mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(93,214,208,.15)",name="Empirical Q90 error range"))
        fig.add_trace(go.Scatter(x=valid["Forecast_Step"],y=valid["Forecast_Care_Load"],mode="lines+markers",line=dict(color="#5dd6d0",width=3),name="Point forecast"))
        fig.update_xaxes(dtick=1,title="Future observed record")
        st.plotly_chart(layout(fig,400),use_container_width=True)
        widths=valid["Upper_Q90"]-valid["Lower_Q90"]
        a,b=st.columns(2); a.metric("Mean Q90 range width",f"{widths.mean():,.0f}"); b.metric("Largest Q90 range width",f"{widths.max():,.0f}")
        st.dataframe(valid,use_container_width=True,hide_index=True)
    st.markdown('<div class="notice">Empirical error ranges are not guaranteed prediction intervals. Nominal coverage is not assured.</div>',unsafe_allow_html=True)

elif page=="Capacity planner":
    st.title("Capacity planner")
    st.markdown('<div class="subtitle">Test a capacity assumption against the selected forecast.</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Scenario capacity",f"{capacity:,.0f}")
    b.metric("Peak forecast",f"{peak:,.0f}")
    c.metric("Peak utilization",f"{peak/capacity*100:.1f}%")
    d.metric("Additional places at peak",f"{int(fc['Estimated_Additional_Places'].max()):,}")
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fc["Forecast_Step"],y=fc["Forecast_Care_Load"],mode="lines+markers",name="Forecast load",line=dict(color="#64a8ff",width=3)))
    fig.add_hline(y=capacity*warning,line_dash="dot",line_color="#5dd6d0",annotation_text=f"Warning · {warning_pct}%")
    fig.add_hline(y=capacity,line_dash="dash",line_color="#f3c77b",annotation_text="Capacity scenario")
    fig.update_xaxes(dtick=1,title="Future observed record")
    st.plotly_chart(layout(fig,380),use_container_width=True)
    st.dataframe(fc[["Forecast_Step","Forecast_Care_Load","Capacity_Utilization_Percent","Estimated_Additional_Places","Scenario_Risk"]].round(2),use_container_width=True,hide_index=True)
    st.markdown('<div class="notice">Capacity is hypothetical unless replaced with a verified, relevant capacity figure.</div>',unsafe_allow_html=True)

elif page=="Scenario lab":
    st.title("Scenario lab")
    st.markdown('<div class="subtitle">Compare analyst-selected demand stress assumptions. These are not probabilistic forecasts.</div>',unsafe_allow_html=True)
    selected=st.multiselect("Scenarios",["Baseline","Elevated_Demand","High_Demand_Stress"],default=["Baseline","Elevated_Demand","High_Demand_Stress"],format_func=lambda x:{"Baseline":"Baseline · 0%","Elevated_Demand":"Elevated demand · +10%","High_Demand_Stress":"High-demand stress · +25%"}[x])
    sub=scenarios[scenarios["Scenario"].isin(selected)].copy()
    if sub.empty: st.info("Choose at least one scenario.")
    else:
        fig=px.line(sub,x="Forecast_Step",y="Projected_Care_Load",color="Scenario",markers=True,labels={"Forecast_Step":"Future observed record","Projected_Care_Load":"Scenario care load"})
        fig.add_hline(y=capacity,line_dash="dash",line_color="#f3c77b",annotation_text="Capacity scenario")
        st.plotly_chart(layout(fig,390),use_container_width=True)
        summary_table=sub.groupby("Scenario",as_index=False).agg(Peak_Load=("Projected_Care_Load","max"),Peak_Utilization=("Utilization_Percent","max"),Max_Additional_Places=("Estimated_Additional_Places","max"))
        st.dataframe(summary_table,use_container_width=True,hide_index=True)
        st.download_button("Download selected scenarios",sub.to_csv(index=False).encode(),file_name="demand_stress_scenarios.csv",mime="text/csv",type="primary")
    st.caption("Scenario multipliers are assumptions, not probabilities estimated from data.")

elif page=="Data explorer":
    st.title("Data explorer")
    st.markdown('<div class="subtitle">Filter observations, inspect values, and export selected rows.</div>',unsafe_allow_html=True)
    dmin=hist["Date"].min().date(); dmax=hist["Date"].max().date()
    dates=st.date_input("Observation date range",value=(dmin,dmax),min_value=dmin,max_value=dmax)
    if isinstance(dates,tuple) and len(dates)==2: start,end=dates
    else: start=end=dates
    sub=hist[hist["Date"].dt.date.between(start,end)].copy()
    a,b,c=st.columns(3)
    a.metric("Records in selection",f"{len(sub):,}")
    b.metric("Mean care load",f"{sub[TARGET].mean():,.0f}" if len(sub) else "—")
    c.metric("Maximum care load",f"{sub[TARGET].max():,.0f}" if len(sub) else "—")
    st.dataframe(sub,use_container_width=True,hide_index=True)
    st.download_button("Download filtered data",sub.to_csv(index=False).encode(),file_name="filtered_care_observations.csv",mime="text/csv",type="primary")
    st.markdown('<div class="notice">Observation dates are irregular. Missing calendar dates are not converted to zero or imputed.</div>',unsafe_allow_html=True)

elif page=="Methodology":
    st.title("Methodology & limitations")
    st.markdown('<div class="subtitle">A concise record of what the dashboard does—and what it does not claim.</div>',unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("Data")
        st.write(f"The processed dataset contains **{len(hist):,} observations**, from **{hist['Date'].min():%d %b %Y}** to **{hist['Date'].max():%d %b %Y}**.")
        st.write("Observation dates are irregular; forecasts refer to future observed records, not calendar days.")
    with st.container(border=True):
        st.subheader("Models & validation")
        st.write("Models: Naive, ARIMA(1,1,1), and Holt Damped Trend. Walk-forward and chronological holdout evaluations are shown separately.")
    with st.container(border=True):
        st.subheader("Uncertainty")
        st.write("Empirical ranges use historical out-of-sample absolute errors by forecast step. They are not guaranteed prediction intervals.")
    with st.container(border=True):
        st.subheader("Capacity scenarios")
        st.write("Capacity is a configurable threshold. +10% and +25% demand scenarios are hypothetical assumptions, not probabilities.")
    with st.container(border=True):
        st.subheader("Intended use")
        st.write("This is an analytical prototype. Do not use it as the sole basis for operational decisions without verified capacity data, domain review, and further validation.")

st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#71839b;font-size:.76rem;"><span>CARE LOAD INTELLIGENCE · Forecasting workspace</span><span>Observed records · Transparent assumptions · Reproducible outputs · Prototype</span></div>',unsafe_allow_html=True)
