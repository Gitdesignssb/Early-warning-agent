import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Early Warning Loan Agent", layout="wide")

st.title("💡 Early Warning Loan Default Agent")
st.caption("Upload a loan repayment CSV, map columns (if needed), and get early risk alerts.")

# -------------------------------
# Helpers
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
    """Robust CSV loader that guesses delimiter and handles UTF-8 BOM."""
    return pd.read_csv(file_or_path, sep=None, engine="python", encoding="utf-8-sig")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower, strip, replace spaces/hyphens with underscores; unify common synonyms."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace("-", "_")
    )
    # Try simple synonym remaps
    remap = {}
    for logical, candidates in REQUIRED_LOGICAL_FIELDS.items():
        for c in candidates:
            if c in df.columns:
                remap[c] = logical  # map first match to canonical name
                break
    if remap:
        df = df.rename(columns=remap)
    return df

def choose_or_confirm_mapping(df: pd.DataFrame):
    """If any canonical field is missing, let user choose via dropdown."""
    cols = list(df.columns)
    st.subheader("🔧 Column Mapping")

    mapping = {}
    for logical, candidates in REQUIRED_LOGICAL_FIELDS.items():
        default = None
        # Suggest the first matched synonym if already renamed
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
        return None  # force user to map

    return mapping

def coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def parse_date(series):
    return pd.to_datetime(series, errors="coerce")

# -------------------------------
# Data input
# -------------------------------
uploaded = st.file_uploader("📤 Upload Loan Repayment CSV", type=["csv"])

if uploaded is not None:
    df_raw = load_csv(uploaded)
else:
    st.info("No file uploaded. Using bundled **sample_data.csv**.")
    df_raw = load_csv("sample_data.csv")

st.subheader("📄 Raw Data Preview")
st.dataframe(df_raw.head(20), use_container_width=True)

# Normalize and map
df = normalize_columns(df_raw)
mapping = choose_or_confirm_mapping(df)
if mapping is None:
    st.stop()

# Create a working frame with canonical names
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

# Coerce types
work["emi_amount"] = coerce_numeric(work["emi_amount"])
work["amount_paid"] = coerce_numeric(work["amount_paid"])
work["emi_due_date"] = parse_date(work["emi_due_date"])
work["payment_date"] = parse_date(work["payment_date"])

# Handle bounce flag (if using return_code etc, anything non-null/non-zero becomes bounce)
if "bounce_flag" in work.columns:
    # Convert strings like 'R01'/'NSF' into 1, blanks into 0
    bf = work["bounce_flag"].copy()
    bf = bf.fillna(0)
    # if it is text (codes), mark as 1 where non-empty
    if bf.dtype == object:
        work["bounce_flag"] = np.where(bf.astype(str).str.strip().eq(""), 0, 1)
    else:
        work["bounce_flag"] = coerce_numeric(bf).fillna(0).astype(int)
else:
    work["bounce_flag"] = 0

# Feature engineering
today = pd.Timestamp(datetime.now().date())
work["emi_gap"] = (work["emi_amount"] - work["amount_paid"]).fillna(0)

# If payment_date missing, days since due date; else (payment - due)
work["days_delay"] = np.where(
    work["payment_date"].isna(),
    (today - work["emi_due_date"]).dt.days,
    (work["payment_date"] - work["emi_due_date"]).dt.days
)
work["days_delay"] = work["days_delay"].fillna(0).astype(int)

# Rule-based risk
work["risk_flag"] = np.where(
    (work["bounce_flag"] == 1) | (work["emi_gap"] > 0) | (work["days_delay"] > 0),
    "High",
    "Low"
)

# Anomaly detection (safe on small data)
can_do_ml = work[["emi_gap", "days_delay"]].dropna().shape[0] >= 10
if can_do_ml:
    features = work[["emi_gap", "days_delay"]].astype(float).fillna(0)
    # Bound contamination to avoid exceptions on smallish datasets
    contamination = min(0.2, max(0.05, 2.0 / len(features)))
    model = IsolationForest(contamination=contamination, random_state=42)
    work["anomaly_score"] = model.fit_predict(features)
    work["anomaly_flag"] = np.where(work["anomaly_score"] == -1, "Anomaly", "Normal")
else:
    work["anomaly_flag"] = "Normal"

# -------------------------------
# Results
# -------------------------------
st.subheader("📊 Processed Portfolio")
st.dataframe(work, use_container_width=True)

st.subheader("🚨 Risk Alerts")
alerts = work[(work["risk_flag"] == "High") | (work["anomaly_flag"] == "Anomaly")][
    ["loan_id", "customer_name", "risk_flag", "anomaly_flag", "emi_gap", "days_delay", "bounce_flag"]
]

c1, c2, c3 = st.columns(3)
c1.metric("Total Loans", len(work))
c2.metric("High-Risk (rules)", int((work["risk_flag"] == "High").sum()))
c3.metric("Anomalies", int((work["anomaly_flag"] == "Anomaly").sum()))

if alerts.empty:
    st.success("No high-risk loans detected right now.")
else:
    st.error("High-risk loans detected!")
    st.dataframe(alerts, use_container_width=True)
    st.download_button("⬇️ Download Risk Report (CSV)", alerts.to_csv(index=False), "risk_report.csv", "text/csv")
