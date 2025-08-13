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
# Page & Env
# -------------------------------
st.set_page_config(page_title="Early Warning Loan Agent", layout="wide")
load_dotenv()  # loads .env if present

st.title("💡 Early Warning Loan Default Agent")
st.caption("Upload portfolio CSV → detect partials/bounces/delays → anomaly detection → severity tiers → charts → simulate or send alerts.")

# -------------------------------
# Config (sidebar controls)
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Severity thresholds**")
    bounce_weight = st.slider("Bounce weight (points)", 10, 80, 50, 5)
    max_gap_points = st.slider("Max points for EMI gap %", 10, 60, 40, 5)
    delay_points_per_day = st.slider("Delay points per day (cap 30)", 1, 5, 2, 1)
    anomaly_points = st.slider("Anomaly points", 5, 30, 20, 1)
    severity_watch = st.slider("WATCH threshold (score ≥)", 10, 80, 30, 5)
    severity_action = st.slider("ACTION threshold (score ≥)", 20, 100, 60, 5)

    st.markdown("---")
    st.markdown("**Anomaly detection**")
    contamination = st.slider("Anomaly rate (contamination)", 0.01, 0.30, 0.20, 0.01)

    st.markdown("---")
    st.markdown("**Alert routing**")
    default_officer_email = st.text_input("Officer email (for demo)", "credit.officer@example.com")
    teams_webhook_env = os.getenv("TEAMS_WEBHOOK_URL", "")
    teams_webhook = st.text_input("Teams Incoming Webhook URL", teams_webhook_env, type="password", help="Optional: paste a Teams incoming webhook URL")
    dry_run = st.checkbox("Dry run (simulate alerts without sending)", value=True)
    st.caption("Tip: keep Dry run checked for Vibe demo; uncheck only if you configured a real Teams webhook and SMTP.")

# -------------------------------
# Helpers: schema & mapping
# -------------------------------
REQUIRED_LOGICAL_FIELDS = {
    "loan_id": ["loan_id", "acct_id", "account_id", "loan number", "loan_no", "id"],
    "customer_name": ["customer_name", "borrower", "name", "customer"],
    "emi_due_date": ["emi_due_date", "due_date", "emi_due", "installment_due_date"],
    "emi_amount": ["emi_amount", "emi amt", "emi", "installment_amount", "scheduled_amount"],
    "amount_paid": ["amount_paid", "paid_amount", "amt_paid", "amount paid", "received_amount"],
    "payment_date": ["payment_date", "paid_on", "posting_date", "posted_date", "payment_dt"],
    "bounce_flag": ["bounce_flag", "bounced", "cheque_bounce", "return_code", "ach_return", "nsf_flag"]
}

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
    if remap:
        df = df.rename(columns=remap)
    return df

def choose_or_confirm_mapping(df: pd.DataFrame):
    cols = list(df.columns)
    st.subheader("🔧 Column Mapping")

    mapping = {}
    for logical, candidates in REQUIRED_LOGICAL_FIELDS.items():
        default = None
        if logical in cols:
            default = logical
        else:
            for c in candidates:
                if c in cols:
                    default = c
                    break
        mapping[logical] = st.selectbox(
            f"Select column for **{logical}**",
            options=["-- not present --"] + cols,
            index=(["-- not present --"] + cols).index(default) if default in cols else 0,
            help=f"Expected synonyms: {', '.join(candidates)}"
        )

    missing = [k for k, v in mapping.items() if v == "-- not present --" and k not in ("payment_date", "bounce_flag", "customer_name")]
    if missing:
        st.warning(
            "The following required fields are not mapped: "
            + ", ".join(missing)
            + ". Please select appropriate columns."
        )
        return None
    return mapping

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_date(series):
    return pd.to_datetime(series, errors="coerce")

# -------------------------------
# Upload or sample
# -------------------------------
uploaded = st.file_uploader("📤 Upload Loan Repayment CSV", type=["csv"])
if uploaded is not None:
    df_raw = load_csv(uploaded)
else:
    st.info("No file uploaded. Using bundled **sample_data.csv**.")
    df_raw = load_csv("sample_data.csv")

st.subheader("📄 Raw Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

# Normalize + mapping
df = normalize_columns(df_raw)
mapping = choose_or_confirm_mapping(df)
if mapping is None:
    st.stop()

def pick(colname):
    sel = mapping[colname]
    return df[sel] if sel != "-- not present --" else pd.Series([np.nan] * len(df))

work = pd.DataFrame({
    "loan_id": pick("loan_id"),
    "customer_name": pick("customer_name"),
    "emi_due_date": pick("emi_due_date"),
    "emi_amount": pick("emi_amount"),
    "amount_paid": pick("amount_paid"),
    "payment_date": pick("payment_date"),
    "bounce_flag": pick("bounce_flag")
})

# Types
work["emi_amount"] = coerce_numeric(work["emi_amount"])
work["amount_paid"] = coerce_numeric(work["amount_paid"])
work["emi_due_date"] = parse_date(work["emi_due_date"])
work["payment_date"] = parse_date(work["payment_date"])

# Bounce normalization
if "bounce_flag" in work.columns:
    bf = work["bounce_flag"].copy().fillna(0)
    if bf.dtype == object:
        work["bounce_flag"] = np.where(bf.astype(str).str.strip().eq(""), 0, 1)
    else:
        work["bounce_flag"] = coerce_numeric(bf).fillna(0).astype(int)
else:
    work["bounce_flag"] = 0

# Features
today = pd.Timestamp(datetime.now().date())
work["emi_gap"] = (work["emi_amount"] - work["amount_paid"]).fillna(0)

work["days_delay"] = np.where(
    work["payment_date"].isna(),
    (today - work["emi_due_date"]).dt.days,
    (work["payment_date"] - work["emi_due_date"]).dt.days
)
work["days_delay"] = work["days_delay"].fillna(0).astype(int)

# Rule-based risk (baseline)
work["rule_high"] = np.where(
    (work["bounce_flag"] == 1) | (work["emi_gap"] > 0) | (work["days_delay"] > 0),
    1, 0
)

# Anomaly detection
can_do_ml = work[["emi_gap", "days_delay"]].dropna().shape[0] >= 10
if can_do_ml:
    features = work[["emi_gap", "days_delay"]].astype(float).fillna(0)
    # Bound contamination for small data
    contamination_eff = min(0.3, max(contamination, 2.0 / len(features)))
    model = IsolationForest(contamination=contamination_eff, random_state=42)
    work["anomaly_score_raw"] = model.fit_predict(features)  # -1 anomaly, 1 normal
    work["anomaly_flag"] = np.where(work["anomaly_score_raw"] == -1, "Anomaly", "Normal")
else:
    work["anomaly_flag"] = "Normal"

# Reason codes
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

# Scoring → Severity tiers
def risk_score(row):
    score = 0.0
    # bounce
    if row["bounce_flag"] == 1:
        score += bounce_weight
    # gap % of EMI
    emi = row["emi_amount"] if pd.notna(row["emi_amount"]) and row["emi_amount"] else 0.0
    gap = max(0.0, float(row["emi_gap"]))
    gap_pct = (gap / emi) if emi > 0 else (1.0 if gap > 0 else 0.0)
    score += min(max_gap_points, gap_pct * max_gap_points)
    # delay (cap 30 points)
    score += min(30, max(0, int(row["days_delay"])) * delay_points_per_day)
    # anomaly
    if row.get("anomaly_flag") == "Anomaly":
        score += anomaly_points
    return round(score, 1)

work["risk_score"] = work.apply(risk_score, axis=1)

def severity(score):
    if score >= severity_action:
        return "Action"
    if score >= severity_watch:
        return "Watch"
    return "Info"

work["severity"] = work["risk_score"].apply(severity)

# -------------------------------
# Dashboard
# -------------------------------
st.subheader("📊 Portfolio Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Loans", len(work))
c2.metric("High-Risk (rules)", int(work["rule_high"].sum()))
c3.metric("Anomalies", int((work["anomaly_flag"] == "Anomaly").sum()))
c4.metric("Action Severity", int((work["severity"] == "Action").sum()))

# Charts
colA, colB = st.columns(2)
with colA:
    fig_sev = px.histogram(work, x="severity", color="severity",
                           category_orders={"severity": ["Info", "Watch", "Action"]},
                           title="Severity Distribution")
    st.plotly_chart(fig_sev, use_container_width=True)
with colB:
    fig_gap = px.histogram(work, x="emi_gap", nbins=20, title="EMI Gap Distribution")
    st.plotly_chart(fig_gap, use_container_width=True)

fig_scatter = px.scatter(
    work, x="days_delay", y="emi_gap", color="severity",
    hover_data=["loan_id", "customer_name", "reason_codes", "risk_score"],
    title="Risk Landscape: Days Delay vs EMI Gap"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Filter + results table
st.subheader("🚨 Alerts")
sev_filter = st.multiselect("Filter by severity", ["Action", "Watch", "Info"], default=["Action", "Watch"])
alerts = work[work["severity"].isin(sev_filter)][
    ["loan_id", "customer_name", "severity", "risk_score", "reason_codes", "emi_gap", "days_delay", "bounce_flag"]
].sort_values(["severity", "risk_score"], ascending=[False, False])

if alerts.empty:
    st.success("No alerts at the selected severity levels.")
else:
    st.dataframe(alerts, use_container_width=True)
    st.download_button("⬇️ Download Risk Report (CSV)", alerts.to_csv(index=False), "risk_report.csv", "text/csv")

# -------------------------------
# Alert routing (simulation + optional real)
# -------------------------------
def build_alert_message(row):
    return (
        f"🚨 Loan Alert: {row['severity']}\n"
        f"Loan: {row['loan_id']} | Customer: {row.get('customer_name','-')}\n"
        f"Score: {row['risk_score']}\n"
        f"Reasons: {row['reason_codes']}\n"
        f"EMI Gap: {row['emi_gap']} | Days Delay: {row['days_delay']} | Bounce: {row['bounce_flag']}"
    )

def send_to_teams(webhook_url: str, text: str) -> bool:
    try:
        payload = {"text": text}
        r = requests.post(webhook_url, json=payload, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False

st.markdown("---")
st.subheader("📣 Send Alerts (Demo)")
st.caption("By default this is a **simulation** for demo purposes. Provide a Teams Webhook and uncheck **Dry run** to actually post to a Teams channel.")

# Choose which alerts to send
send_scope = st.selectbox("Which alerts to send?", ["Action only", "Action + Watch", "All severities"])
if send_scope == "Action only":
    to_send = alerts[alerts["severity"] == "Action"]
elif send_scope == "Action + Watch":
    to_send = alerts[alerts["severity"].isin(["Action", "Watch"])]
else:
    to_send = alerts.copy()

if st.button("Send / Simulate Alerts"):
    if to_send.empty:
        st.info("No alerts to send based on your filter/scope.")
    else:
        sent = 0
        sim_preview = []
        for _, row in to_send.iterrows():
            msg = build_alert_message(row)
            sim_preview.append(msg)

            if not dry_run and teams_webhook:
                ok = send_to_teams(teams_webhook, msg)
                if ok:
                    sent += 1
            # (Optional) Email demo: often blocked on hosted envs, so we skip real SMTP
            # You could add smtplib here if your environment allows outbound SMTP.

        if dry_run or not teams_webhook:
            st.warning("Dry run ON or no Teams webhook provided → **simulation only**.")
            with st.expander("Preview of messages"):
                for m in sim_preview[:10]:
                    st.code(m)
                if len(sim_preview) > 10:
                    st.write(f"...and {len(sim_preview) - 10} more.")

        if not dry_run and teams_webhook:
            st.success(f"Posted {sent} alert(s) to Teams.")
