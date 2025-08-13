import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import plotly.express as px
import requests
from dotenv import load_dotenv

# -------------------------------
# App & Env
# -------------------------------
st.set_page_config(page_title="Early Warning Loan Agent", layout="wide")
load_dotenv()

st.title("💡 Early Warning Loan Default Agent")
st.caption(
    "Upload CSV → auto‑map behind the scenes → detect partials/bounces/delays → anomaly detection → "
    "risk score & severity tiers → rich portfolio views → customer drilldowns → simulate/send alerts."
)

# -------------------------------
# Sidebar controls (Condensed)
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    section = st.selectbox(
        "Choose Configuration Section",
        ["Severity thresholds", "Anomaly detection", "Alert routing"]
    )

    if section == "Severity thresholds":
        bounce_weight = st.slider("Bounce weight (points)", 10, 80, 50, 5)
        max_gap_points = st.slider("Max points for EMI gap %", 10, 60, 40, 5)
        delay_points_per_day = st.slider("Delay points/day (cap 30)", 1, 5, 2, 1)
        anomaly_points = st.slider("Anomaly points", 5, 30, 20, 1)
        severity_watch = st.slider("WATCH threshold (score ≥)", 10, 80, 30, 5)
        severity_action = st.slider("ACTION threshold (score ≥)", 20, 100, 60, 5)

    elif section == "Anomaly detection":
        contamination = st.slider("Anomaly rate (contamination)", 0.01, 0.30, 0.20, 0.01)
