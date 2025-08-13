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
    "risk score & severity tiers → rich portfolio views → simulate/send alerts."
)

# -------------------------------
# Sidebar controls
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

    elif section == "Alert routing":
        default_officer_email = st.text_input("Officer email (demo)", "credit.officer@example.com")
        teams_webhook_env = os.getenv("TEAMS_WEBHOOK_URL", "")
        teams_webhook = st.text_input("Teams Incoming Webhook URL", teams_webhook_env, type="password")
        dry_run = st.checkbox("Dry run (simulate only)", value=True)

# -------------------------------
# Helpers & Schema
# -------------------------------
REQUIRED_LOGICAL_FIELDS = {
    "loan_id":       ["loan_id", "acct_id", "account_id", "loan number", "loan_no", "id"],
    "customer_name": ["customer_name", "borrower", "name", "customer"],
    "emi_due_date":  ["emi_due_date", "due_date", "emi_due", "installment_due_date"],
    "emi_amount":    ["emi_amount", "emi amt", "emi", "installment_amount", "scheduled_amount"],
    "amount_paid":   ["amount_paid", "paid_amount", "amt_paid", "amount paid", "received_amount"],
    "payment_date":  ["payment_date", "paid_on", "posting_date", "posted_date", "payment_dt"],
    "bounce_flag":   ["bounce_flag", "bounced", "cheque_bounce", "return_code", "ach_return", "nsf_flag"],
    "loan_type":     ["loan_type", "product", "product_type", "loan_product", "segment"]
}
REQUIRED_MIN = ["loan_id", "emi_due_date", "emi_amount", "amount_paid"]

@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    return pd.read_csv(file_or_path, sep=None, engine="python", encoding="utf-8-sig")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace("-", "_")
    )
    remap = {}
    for logical, candidates in REQUIRED_LOGICAL_FIELDS.items():
        for c in candidates:
            if c in df.columns:
                remap[c] = logical
                break
    return df.rename(columns=remap) if remap else df

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_date(series):
    return pd.to_datetime(series, errors="coerce")

def build_alert_message(row):
    return (
        f"🚨 Loan Alert: {row['severity']}\n"
        f"Loan: {row['loan_id']} | Customer: {row.get('customer_name','-')}\n"
        f"Loan Type: {row.get('loan_type','Unknown')}\n"
        f"Score: {row['risk_score']}\n"
        f"Reasons: {row['reason_codes']}\n"
        f"EMI Gap: {row['emi_gap']:.0f} | Gap%: {row['emi_gap_pct']:.0%} | "
        f"Days Delay: {row['days_delay']} | Bounce: {int(row['bounce_flag'])}"
    )

def send_to_teams(webhook_url: str, text: str) -> bool:
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False

def severity_from_score(score, watch, action):
    if score >= action:
        return "Action"
    if score >= watch:
        return "Watch"
    return "Info"

# -------------------------------
# Tabs
# -------------------------------
tab_data, tab_portfolio, tab_risk, tab_alerts, tab_notify = st.tabs(
    ["📁 Data", "📊 Portfolio", "🗺️ Risk Landscape", "🚨 Alerts", "📣 Notify"]
)

# ============ DATA TAB ============
with tab_data:
    st.subheader("📄 Raw Data")

    uploaded = st.file_uploader("Upload Loan Repayment CSV", type=["csv"])
    if uploaded is not None:
        df_raw = load_csv(uploaded)
    else:
        st.info("No file uploaded. Please upload a CSV file.")
        st.stop()

    st.dataframe(df_raw.head(50), use_container_width=True, height=220)

    df_norm = normalize_columns(df_raw)

    missing_required = [col for col in REQUIRED_MIN if col not in df_norm.columns]
    if missing_required:
        st.error(
            "Your file is missing required columns after auto-mapping: "
            + ", ".join(missing_required)
            + ".\n\nExpected fields include: "
            + ", ".join(REQUIRED_MIN)
            + "."
        )
        st.stop()

    def pick(colname, default=np.nan):
        return df_norm[colname] if colname in df_norm.columns else pd.Series([default] * len(df_norm))

    work = pd.DataFrame({
        "loan_id":       pick("loan_id"),
        "customer_name": pick("customer_name", default="Unknown"),
        "loan_type":     pick("loan_type", default="Unknown"),
        "emi_due_date":  pick("emi_due_date"),
        "emi_amount":    pick("emi_amount"),
        "amount_paid":   pick("amount_paid"),
        "payment_date":  pick("payment_date"),
        "bounce_flag":   pick("bounce_flag", default=0),
    })

    work["emi_amount"] = coerce_numeric(work["emi_amount"])
    work["amount_paid"] = coerce_numeric(work["amount_paid"])
    work["emi_due_date"] = parse_date(work["emi_due_date"])
    work["payment_date"] = parse_date(work["payment_date"])

    bf = work["bounce_flag"].copy().fillna(0)
    if bf.dtype == object:
        work["bounce_flag"] = np.where(bf.astype(str).str.strip().eq(""), 0, 1)
    else:
        work["bounce_flag"] = coerce_numeric(bf).fillna(0).astype(int)

    today = pd.Timestamp(datetime.now().date())
    work["emi_gap"] = (work["emi_amount"] - work["amount_paid"]).fillna(0)

    work["days_delay"] = np.where(
        work["payment_date"].isna(),
        (today - work["emi_due_date"]).dt.days,
        (work["payment_date"] - work["emi_due_date"]).dt.days
    )
    work["days_delay"] = work["days_delay"].fillna(0).astype(int)

    work["emi_gap_pct"] = np.where(
        (work["emi_amount"] > 0) & (work["emi_gap"] > 0),
        work["emi_gap"] / work["emi_amount"],
        0.0
    )

    work["rule_high"] = np.where(
        (work["bounce_flag"] == 1) | (work["emi_gap"] > 0) | (work["days_delay"] > 0),
        1, 0
    )

    can_do_ml = work[["emi_gap", "days_delay"]].dropna().shape[0] >= 10
    if can_do_ml:
        features = work[["emi_gap", "days_delay"]].astype(float).fillna(0)
        contamination_eff = min(0.3, max(contamination, 2.0 / len(features)))
        model = IsolationForest(contamination=contamination_eff, random_state=42)
        work["anomaly_score_raw"] = model.fit_predict(features)
        work["anomaly_flag"] = np.where(work["anomaly_score_raw"] == -1, "Anomaly", "Normal")
    else:
        work["anomaly_flag"] = "Normal"

    def reasons(row):
        r = []
        if row.get("bounce_flag", 0) == 1:
            r.append("Bounced payment")
        if pd.notna(row.get("amount_paid")) and pd.notna(row.get("emi_amount")):
            if float(row["amount_paid"]) == 0:
                r.append("Missed payment")
            elif float(row["amount_paid"]) < float(row["emi_amount"]):
                r.append("Partial payment")
        if row.get("days_delay", 0) > 0:
            r.append(f"Delayed by {int(row['days_delay'])} days")
        if row.get("anomaly_flag") == "Anomaly":
            r.append("Anomalous pattern")
        if not r:
            r.append("No issues")
        return ", ".join(r)

    work["reason_codes"] = work.apply(reasons, axis=1)

    def risk_score(row):
        score = 0.0
        if row["bounce_flag"] == 1:
            score += bounce_weight
        gap_pct = float(row["emi_gap_pct"])
        score += min(max_gap_points, gap_pct * max_gap_points)
        score += min(30, max(0, int(row["days_delay"])) * delay_points_per_day)
        if row.get("anomaly_flag") == "Anomaly":
            score += anomaly_points
        return round(score, 1)

    work["risk_score"] = work.apply(risk_score, axis=1)
    work["severity"] = work["risk_score"].apply(lambda s: severity_from_score(s, severity_watch, severity_action))

    st.success("Data ready ✅")

# ============ PORTFOLIO TAB ============
with tab_portfolio:
    st.subheader("📊 Portfolio Overview")

    total = len(work)
    rules_high = int(work["rule_high"].sum())
    anomalies = int((work["anomaly_flag"] == "Anomaly").sum())
    actions = int((work["severity"] == "Action").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Loans", total)
    m2.metric("High‑Risk (rules)", rules_high)
    m3.metric("Anomalies", anomalies)
    m4.metric("Action Severity", actions)

    cA, cB = st.columns(2)
    fig_sev = px.histogram(
        work, x="severity", color="severity",
        color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
        category_orders={"severity":["Info","Watch","Action"]},
        title="Severity Distribution"
    )
    cA.plotly_chart(fig_sev, use_container_width=True)

    fig_stack = px.histogram(
        work, x="loan_type", color="severity", barmode="group",
        color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
        title="Loan Types by Severity"
    )
    cB.plotly_chart(fig_stack, use_container_width=True)

# ============ RISK LANDSCAPE TAB ============
with tab_risk:
    st.subheader("🗺️ Risk Landscape")

    view = st.radio("Choose view", ["Quadrant (Gap% vs Days Delay)", "Density Heatmap", "Sunburst (Loan Type → Severity)"], horizontal=True)

    if view == "Quadrant (Gap% vs Days Delay)":
        fig = px.scatter(
            work,
            x="days_delay", y="emi_gap_pct",
            color="severity",
            color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
            size="risk_score",
            hover_data=["loan_id","customer_name","loan_type","reason_codes","risk_score"],
            title="Quadrant: Days Delay vs EMI Gap % (bubble size = risk score)"
        )
        fig.update_yaxes(tickformat=".0%", title_text="EMI Gap (%)")
        fig.update_xaxes(title_text="Days Delay")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Density Heatmap":
        fig = px.density_heatmap(
            work, x="days_delay", y="emi_gap_pct",
            nbinsx=20, nbinsy=20,
            color_continuous_scale="YlOrRd",
            title="Density Heatmap: Portfolio Hotspots (Days Delay vs EMI Gap %)"
        )
        fig.update_yaxes(tickformat=".0%", title_text="EMI Gap (%)")
        fig.update_xaxes(title_text="Days Delay")
        st.plotly_chart(fig, use_container_width=True)

    else:
        sb_df = work.copy()
        sb_df["count"] = 1
        fig = px.sunburst(
            sb_df, path=["loan_type", "severity"], values="count",
            color="severity",
            color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
            title="Sunburst: Composition by Loan Type → Severity"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============ ALERTS TAB ============
with tab_alerts:
    st.subheader("🚨 Alerts")

    sev_filter = st.multiselect("Filter by severity", ["Action","Watch","Info"], default=["Action","Watch"])
    alerts = work[work["severity"].isin(sev_filter)][
        ["loan_id","customer_name","loan_type","severity","risk_score","reason_codes",
         "emi_gap","emi_gap_pct","days_delay","bounce_flag"]
    ].sort_values(["severity","risk_score"], ascending=[False,False])

    if alerts.empty:
        st.success("No alerts at the selected severity levels.")
    else:
        display = alerts.copy()
        display["emi_gap_pct"] = display["emi_gap_pct"].map(lambda v: f"{v:.0%}")
        st.dataframe(display, use_container_width=True, height=320)
        st.download_button("⬇️ Download Risk Report (CSV)", alerts.to_csv(index=False), "risk_report.csv", "text/csv")

# ============ NOTIFY TAB ============
with tab_notify:
    st.subheader("📣 Send Alerts")

    base = work[work["severity"].isin(["Action","Watch"])]
    def msg_df(df):
        return df[["loan_id","customer_name","loan_type","severity","risk_score","reason_codes",
                   "emi_gap","emi_gap_pct","days_delay","bounce_flag"]].copy()

    scope = st.selectbox("Which alerts to send?", ["Action only", "Action + Watch", "All severities"])
    if scope == "Action only":
        to_send = msg_df(work[work["severity"] == "Action"])
    elif scope == "Action + Watch":
        to_send = msg_df(base)
    else:
        to_send = msg_df(work)

    with st.expander("Preview first 5 messages", expanded=True):
        for _, row in to_send.head(5).iterrows():
            st.code(build_alert_message(row), language=None)

    if st.button("Send / Simulate Alerts"):
        if to_send.empty:
            st.info("No alerts to send for current scope.")
        else:
            st.toast(f"Prepared {len(to_send)} alert(s).", icon="✅")
            sent = 0
            if not dry_run and teams_webhook:
                for _, row in to_send.iterrows():
                    ok = send_to_teams(teams_webhook, build_alert_message(row))
                    if ok:
                        sent += 1
                if sent:
                    st.success(f"Posted {sent} alert(s) to Teams.")
                else:
                    st.warning("Attempted to post alerts but none succeeded. Check webhook/permissions.")
            else:
                with st.expander("Simulation output"):
                    for _, row in to_send.iterrows():
                        st.write(f"📧 {default_officer_email}")
                        st.code(build_alert_message(row), language=None)
                st.warning("Dry run ON or no Teams webhook provided → simulation only.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr style='border:0;height:1px;background:#eee;margin:1.5rem 0;'>", unsafe_allow_html=True)
st.caption("Prototype • Hidden auto‑mapping • Loan type segmentation • Risk Landscape views • Alert simulation")
