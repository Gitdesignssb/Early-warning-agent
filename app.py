import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(page_title="Early Warning Loan Agent", layout="wide")

st.title("💡 Early Warning Loan Default Agent")
st.write("Monitor loan repayments, detect anomalies, and flag early risk signals.")

# File upload or sample data
uploaded_file = st.file_uploader("Upload Loan Repayment CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using sample data...")
    df = pd.read_csv("sample_data.csv")

# Display data
st.subheader("Loan Portfolio Data")
st.dataframe(df)

# Feature engineering
df['emi_gap'] = df['emi_amount'] - df['amount_paid']
df['days_delay'] = df.apply(lambda x: (datetime.now() - pd.to_datetime(x['emi_due_date'])).days 
                             if pd.isnull(x['payment_date']) else 
                             (pd.to_datetime(x['payment_date']) - pd.to_datetime(x['emi_due_date'])).days, axis=1)

# Risk rules
df['risk_flag'] = df.apply(lambda x: 'High' if x['bounce_flag']==1 or x['emi_gap']>0 else 'Low', axis=1)

# Anomaly detection
features = df[['emi_gap','days_delay']].fillna(0)
model = IsolationForest(contamination=0.2, random_state=42)
df['anomaly_score'] = model.fit_predict(features)
df['anomaly_flag'] = df['anomaly_score'].apply(lambda x: 'Anomaly' if x==-1 else 'Normal')

# Display alerts
st.subheader("🚨 Risk Alerts")
alerts = df[(df['risk_flag']=='High') | (df['anomaly_flag']=='Anomaly')]
if not alerts.empty:
    st.error("High Risk Loans Detected!")
    st.dataframe(alerts[['loan_id','customer_name','risk_flag','anomaly_flag','emi_gap','days_delay']])
else:
    st.success("No high-risk loans detected.")

# Download option
st.download_button("Download Risk Report", alerts.to_csv(index=False), "risk_report.csv", "text/csv")
