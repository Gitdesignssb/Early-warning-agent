# app.py
# Early Warning Loan Default Agent (Vibe prototype)
# - Tabs: Data ▸ Portfolio ▸ Alerts ▸ Notify
# - Compact column mapping (popover/expander) with auto-map + presets
# - Features: EMI gap, days delay, bounce normalization
# - Anomaly detection: IsolationForest
# - Risk scoring with configurable thresholds; severity tiers (Info/Watch/Action)
# - Charts: severity distribution, gap histogram, scatter landscape
# - Alert simulation + optional real Teams webhook posting
# - Small UX: toasts, progress, compact tables

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
# App config & env
# -------------------------------
st.set_page_config(page_title="Early Warning Loan Agent", layout="wide")
load_dotenv()  # reads .env if present

st.title("💡 Early Warning Loan Default Agent")
st.caption(
    "Upload portfolio CSV → detect partials/bounces/delays → anomaly detection → "
    "risk score & severity tiers → charts → simulate or send alerts to Teams."
)

# -------------------------------
# Sidebar controls (organized & compact)
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("Severity thresholds", expanded=True):
        bounce_weight = st.slider("Bounce weight (points)", 10, 80, 50, 5)
        max_gap_points = st.slider("Max points for EMI gap %", 10, 60, 40, 5)
        delay_points_per_day = st.slider("Delay points per day (cap 30)", 1, 5, 2, 1)
        anomaly_points = st.slider("Anomaly points", 5, 30, 20, 1)
        severity_watch = st.slider("WATCH threshold (score ≥)", 10, 80, 30, 5)
        severity_action = st.slider("ACTION threshold (score ≥)", 20, 100, 60, 5)

    with st.expander("Anomaly detection", expanded=False):
        contamination = st.slider("Anomaly rate (contamination)", 0.01, 0.30, 0.20, 0.01)

    with st.expander("Alert routing", expanded=False):
        default_officer_email = st.text_input("Officer email (demo)", "credit.officer@example.com")
        teams_webhook_env = os.getenv("TEAMS_WEBHOOK_URL", "")
        teams_webhook = st.text_input(
            "Teams Incoming Webhook URL",
            teams_webhook_env,
            type="password",
            help="Optional: paste a Teams incoming webhook URL"
        )
        dry_run = st.checkbox("Dry run (simulate only)", value=True)

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
REQUIRED_MIN = ["loan_id", "emi_due_date", "emi_amount", "amount_paid"]

@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    # auto-detect delimiter, handle BOM
    return pd.read_csv(file_or_path, sep=None, engine="python", encoding="utf-8-sig")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_")
    )
    # Best-effort synonym remap to canonical names
    remap = {}
    for logical, candidates in REQUIRED_LOGICAL_FIELDS.items():
        for c in candidates:
            if c in df.columns:
                remap[c] = logical
                break
    if remap:
        df = df.rename(columns=remap)
    return df

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_date(series):
    return pd.to_datetime(series, errors="coerce")

def build_alert_message(row):
    return (
        f"🚨 Loan Alert: {row['severity']}\n"
        f"Loan: {row['loan_id']} | Customer: {row.get('customer_name','-')}\n"
        f"Score: {row['risk_score']}\n"
        f"Reasons: {row['reason_codes']}\n"
        f"EMI Gap: {row['emi_gap']} | Days Delay: {row['days_delay']} | Bounce: {int(row['bounce_flag'])}"
    )

def send_to_teams(webhook_url: str, text: str) -> bool:
    try:
        r = requests.post(webhook_url, json={"text": text}, timeout=10)
        return r.status_code in (200, 204)
    except Exception:
        return False

# -------------------------------
# Tabs (Data ▸ Portfolio ▸ Alerts ▸ Notify)
# -------------------------------
tab_data, tab_portfolio, tab_alerts, tab_notify = st.tabs(["📁 Data", "📊 Portfolio", "🚨 Alerts", "📣 Notify"])

# We’ll compute the processed 'work' dataframe inside the Data tab so
# subsequent tabs can reuse it (script runs top-to-bottom).
with tab_data:
    st.subheader("📄 Raw Data")

    uploaded = st.file_uploader("Upload Loan Repayment CSV", type=["csv"])
    if uploaded is not None:
        df_raw = load_csv(uploaded)
    else:
        st.info("No file uploaded. Using bundled **sample_data.csv**.")
        df_raw = load_csv("sample_data.csv")

    st.dataframe(df_raw.head(50), use_container_width=True, height=220)

    # Normalize & attempt auto-mapping
    df_norm = normalize_columns(df_raw)

    auto_map = {k: next((c for c in REQUIRED_LOGICAL_FIELDS[k] if c in df_norm.columns), None)
                for k in REQUIRED_LOGICAL_FIELDS}
    ok_count = sum(1 for k in REQUIRED_LOGICAL_FIELDS if auto_map.get(k))
    auto_ok = all(auto_map.get(k) for k in REQUIRED_MIN)

    st.caption(
        f"🧭 Auto‑mapped **{ok_count}/{len(REQUIRED_LOGICAL_FIELDS)}** fields"
        + (" — looks good!" if auto_ok else " — review recommended")
    )

    # Minimal mapping UI: popover if available, else toggle + expander
    mapping = None
    use_popover = hasattr(st, "popover")
    if use_popover:
        pop = st.popover("Map columns")
        with pop:
            st.write("Map only if needed. Unmapped optional fields can be left blank.")
            cols = list(df_norm.columns)
            with st.form("map_form"):
                mapping = {}
                for logical in REQUIRED_LOGICAL_FIELDS.keys():
                    default = auto_map.get(logical) if auto_map.get(logical) in cols else "-- not present --"
                    mapping[logical] = st.selectbox(
                        f"{logical}",
                        options=["-- not present --"] + cols,
                        index=(["-- not present --"] + cols).index(default) if default in cols else 0,
                    )
                submitted = st.form_submit_button("Apply mapping")
                if not submitted:
                    mapping = None
    else:
        open_mapping = st.toggle("Show mapping", value=not auto_ok, help="Only needed if auto‑mapping is wrong")
        if open_mapping:
            with st.expander("Column Mapping", expanded=not auto_ok):
                st.write("Map only if needed. Unmapped optional fields can be left blank.")
                cols = list(df_norm.columns)
                with st.form("map_form"):
                    mapping = {}
                    for logical in REQUIRED_LOGICAL_FIELDS.keys():
                        default = auto_map.get(logical) if auto_map.get(logical) in cols else "-- not present --"
                        mapping[logical] = st.selectbox(
                            f"{logical}",
                            options=["-- not present --"] + cols,
                            index=(["-- not present --"] + cols).index(default) if default in cols else 0,
                        )
                    submitted = st.form_submit_button("Apply mapping")
                    if not submitted:
                        mapping = None

    # Use mapping if user submitted; else use auto (with optional blanks)
    if mapping is None:
        mapping = {k: (auto_map.get(k) or "-- not present --") for k in REQUIRED_LOGICAL_FIELDS.keys()}

    # Presets (session-scoped)
    if "map_presets" not in st.session_state:
        st.session_state["map_presets"] = {}
    c1, c2 = st.columns(2)
    with c1:
        preset_name = st.text_input("Save mapping preset as", placeholder="e.g., CoreBank v3")
        if st.button("Save preset") and preset_name.strip():
            st.session_state["map_presets"][preset_name.strip()] = mapping.copy()
            st.success(f"Saved preset: {preset_name}")
    with c2:
        if st.session_state["map_presets"]:
            pick_preset = st.selectbox("Load preset", list(st.session_state["map_presets"].keys()))
            if st.button("Use preset"):
                mapping = st.session_state["map_presets"][pick_preset].copy()
                st.success(f"Loaded preset: {pick_preset}")

    # Validate required fields
    missing = [k for k in REQUIRED_MIN if mapping.get(k, "-- not present --") == "-- not present --"]
    if missing:
        st.error(f"Missing required fields: {', '.join(missing)}. Open **Map columns** to fix.")
        st.stop()

    # Build working DF from mapping
    def pick(colname):
        sel = mapping[colname]
        return df_norm[sel] if sel != "-- not present --" else pd.Series([np.nan] * len(df_norm))

    work = pd.DataFrame({
        "loan_id": pick("loan_id"),
        "customer_name": pick("customer_name"),
        "emi_due_date": pick("emi_due_date"),
        "emi_amount": pick("emi_amount"),
        "amount_paid": pick("amount_paid"),
        "payment_date": pick("payment_date"),
        "bounce_flag": pick("bounce_flag"),
    })

    # Types & normalization
    work["emi_amount"] = coerce_numeric(work["emi_amount"])
    work["amount_paid"] = coerce_numeric(work["amount_paid"])
    work["emi_due_date"] = parse_date(work["emi_due_date"])
    work["payment_date"] = parse_date(work["payment_date"])

    bf = work["bounce_flag"].copy() if "bounce_flag" in work.columns else pd.Series([0] * len(work))
    bf = bf.fillna(0)
    if bf.dtype == object:
        work["bounce_flag"] = np.where(bf.astype(str).str.strip().eq(""), 0, 1)
    else:
        work["bounce_flag"] = coerce_numeric(bf).fillna(0).astype(int)

    # Feature engineering
    today = pd.Timestamp(datetime.now().date())
    work["emi_gap"] = (work["emi_amount"] - work["amount_paid"]).fillna(0)

    work["days_delay"] = np.where(
        work["payment_date"].isna(),
        (today - work["emi_due_date"]).dt.days,
        (work["payment_date"] - work["emi_due_date"]).dt.days
    )
    work["days_delay"] = work["days_delay"].fillna(0).astype(int)

    # Simple rule flag
    work["rule_high"] = np.where(
        (work["bounce_flag"] == 1) | (work["emi_gap"] > 0) | (work["days_delay"] > 0),
        1, 0
    )

    # Anomaly detection (safe for small data)
    can_do_ml = work[["emi_gap", "days_delay"]].dropna().shape[0] >= 10
    if can_do_ml:
        features = work[["emi_gap", "days_delay"]].astype(float).fillna(0)
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

    # Risk score & severity
    def risk_score(row):
        score = 0.0
        if row["bounce_flag"] == 1:
            score += bounce_weight
        emi = row["emi_amount"] if pd.notna(row["emi_amount"]) and row["emi_amount"] else 0.0
        gap = max(0.0, float(row["emi_gap"]))
        gap_pct = (gap / emi) if emi > 0 else (1.0 if gap > 0 else 0.0)
        score += min(max_gap_points, gap_pct * max_gap_points)
        score += min(30, max(0, int(row["days_delay"])) * delay_points_per_day)
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

    st.success("Data ready ✅")

# -------------------------------
# Portfolio tab
# -------------------------------
with tab_portfolio:
    st.subheader("📊 Portfolio Summary")

    total = len(work)
    rules_high = int(work["rule_high"].sum())
    anomalies = int((work["anomaly_flag"] == "Anomaly").sum())
    actions = int((work["severity"] == "Action").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Loans", total)
    m2.metric("High‑Risk (rules)", rules_high)
    m3.metric("Anomalies", anomalies)
    m4.metric("Action Severity", actions)

    # Charts
    fig_sev = px.histogram(
        work, x="severity", color="severity",
        color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
        category_orders={"severity":["Info","Watch","Action"]},
        title="Severity Distribution"
    )
    fig_gap = px.histogram(work, x="emi_gap", nbins=20, title="EMI Gap Distribution",
                           color_discrete_sequence=["#1f77b4"])

    cA, cB = st.columns(2)
    cA.plotly_chart(fig_sev, use_container_width=True, key="sev_hist")
    cB.plotly_chart(fig_gap, use_container_width=True, key="gap_hist")

    fig_scatter = px.scatter(
        work, x="days_delay", y="emi_gap", color="severity",
        color_discrete_map={"Info":"#6c757d","Watch":"#f39c12","Action":"#e74c3c"},
        hover_data=["loan_id","customer_name","reason_codes","risk_score"],
        title="Risk Landscape: Days Delay vs EMI Gap"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Drilldown panel
    st.markdown("### 🔎 Drill into a loan")
    left, right = st.columns([1,2])
    with left:
        sel_id = st.selectbox("Pick a loan", work["loan_id"].astype(str).unique())
        focus = work[work["loan_id"].astype(str) == str(sel_id)].iloc[0]
        st.metric("Risk Score", focus["risk_score"])
        pct = min(1.0, focus["risk_score"] / max(1, severity_action))
        st.progress(pct, text=f"Severity threshold (Action ≥ {severity_action})")
    with right:
        st.write(f"**Customer:** {focus.get('customer_name','-')}")
        st.write(f"**Severity:** {focus['severity']}")
        st.write(f"**Reasons:** {focus['reason_codes']}")
        st.write(f"**EMI Gap:** {focus['emi_gap']} | **Days Delay:** {focus['days_delay']} | **Bounce:** {int(focus['bounce_flag'])}")

# -------------------------------
# Alerts tab
# -------------------------------
with tab_alerts:
    st.subheader("🚨 Alerts")

    sev_filter = st.multiselect("Filter by severity", ["Action","Watch","Info"], default=["Action","Watch"])
    severity_emoji = {"Action":"🔴", "Watch":"🟠", "Info":"⚪"}
    alerts = work[work["severity"].isin(sev_filter)][
        ["loan_id","customer_name","severity","risk_score","reason_codes","emi_gap","days_delay","bounce_flag"]
    ].sort_values(["severity","risk_score"], ascending=[False,False]).copy()
    alerts.insert(2, "sev", alerts["severity"].map(severity_emoji))
    alerts = alerts.rename(columns={"sev": " "})  # tiny icon column

    if alerts.empty:
        st.success("No alerts at the selected severity levels.")
    else:
        st.dataframe(
            alerts,
            use_container_width=True,
            height=300
        )
        st.download_button("⬇️ Download Risk Report (CSV)", alerts.to_csv(index=False), "risk_report.csv", "text/csv")

# -------------------------------
# Notify tab (simulate/post alerts)
# -------------------------------
with tab_notify:
    st.subheader("📣 Send Alerts")

    # Use the latest filtered alerts if user just visited Alerts; otherwise rebuild "alerts" defaulting to Action+Watch
    if 'alerts' not in locals() or alerts is None or alerts.empty:
        base = work[work["severity"].isin(["Action","Watch"])]
        alerts = base[["loan_id","customer_name","severity","risk_score","reason_codes","emi_gap","days_delay","bounce_flag"]] \
            .sort_values(["severity","risk_score"], ascending=[False,False])

    send_scope = st.selectbox("Which alerts to send?", ["Action only","Action + Watch","All severities"])
    if send_scope == "Action only":
        to_send = alerts[alerts["severity"]=="Action"]
    elif send_scope == "Action + Watch":
        to_send = alerts[alerts["severity"].isin(["Action","Watch"])]
    else:
        to_send = work[["loan_id","customer_name","severity","risk_score","reason_codes","emi_gap","days_delay","bounce_flag"]].copy()

    # Preview first few
    with st.expander("Preview first 5 messages"):
        for _, row in to_send.head(5).iterrows():
            st.code(build_alert_message(row), language=None)

    if st.button("Send / Simulate Alerts"):
        if to_send.empty:
            st.info("No alerts to send based on your current filter/scope.")
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
                    st.warning("Tried to post alerts but received no success responses. Check webhook URL/permissions.")
            else:
                with st.expander("Simulation output"):
                    for _, row in to_send.iterrows():
                        st.write(f"📧 {default_officer_email}")
                        st.code(build_alert_message(row), language=None)
                st.warning("Dry run ON or no Teams webhook provided → simulation only.")

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<hr style='border: 0; height: 1px; background: #eee; margin: 1.5rem 0;'>",
    unsafe_allow_html=True
)
st.caption(
    "Prototype for Vibe Coding Contest • Features: auto‑mapping, anomaly detection, severity tiers, charts, alert simulation."
)
