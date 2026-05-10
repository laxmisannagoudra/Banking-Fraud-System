"""
Fraud Detection Dashboard - Minimal Working Version
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Real-Time Fraud Detection System")
st.markdown("---")

# API URL
api_url = "http://localhost:8000"

# Check API health
try:
    response = requests.get(f"{api_url}/health", timeout=2)
    if response.status_code == 200:
        st.success("✅ API is connected and ready")
        api_healthy = True
    else:
        st.error("❌ API error")
        api_healthy = False
except:
    st.error("❌ API not running. Start API first: python src/fraud_api.py")
    api_healthy = False

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/credit_card_fraud_10k.csv')
        return df
    except:
        # Create sample data
        np.random.seed(42)
        return pd.DataFrame({
            'amount': np.random.exponential(100, 10000) + 10,
            'merchant_category': np.random.choice(['Electronics', 'Travel', 'Clothing', 'Food', 'Grocery'], 10000),
            'device_trust_score': np.random.randint(0, 100, 10000),
            'is_fraud': np.random.choice([0, 1], 10000, p=[0.985, 0.015])
        })

df = load_data()

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    st.metric("Fraud Cases", f"{df['is_fraud'].sum():,}")
with col3:
    st.metric("Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
with col4:
    st.metric("Avg Amount", f"${df['amount'].mean():.2f}")

st.markdown("---")

# Test Transaction Form
st.subheader("🔍 Test Transaction")

with st.form("test_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tx_id = st.text_input("Transaction ID", f"TX_{datetime.now().strftime('%H%M%S')}")
        amount = st.number_input("Amount ($)", min_value=0.01, value=150.00)
        hour = st.slider("Hour", 0, 23, 14)
        merchant = st.selectbox("Merchant", ['Electronics', 'Travel', 'Clothing', 'Food', 'Grocery'])
    
    with col2:
        foreign = st.selectbox("Foreign Transaction", [0, 1], format_func=lambda x: "Yes" if x else "No")
        location_mismatch = st.selectbox("Location Mismatch", [0, 1], format_func=lambda x: "Yes" if x else "No")
        device_score = st.slider("Device Trust Score", 0, 100, 75)
        velocity = st.number_input("Transactions in 24h", min_value=0, value=2)
        age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35)
    
    submitted = st.form_submit_button("Check Fraud", use_container_width=True)

if submitted and api_healthy:
    payload = {
        "transaction_id": tx_id,
        "amount": amount,
        "transaction_hour": hour,
        "merchant_category": merchant,
        "foreign_transaction": foreign,
        "location_mismatch": location_mismatch,
        "device_trust_score": device_score,
        "velocity_last_24h": velocity,
        "cardholder_age": age
    }
    
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            
            if result['is_fraud']:
                st.error(f"🚨 FRAUD ALERT! Probability: {result['fraud_probability']*100:.1f}%")
            else:
                st.success(f"✅ Legitimate Transaction. Risk: {result['risk_score']:.0f}/100")
            
            st.metric("Fraud Probability", f"{result['fraud_probability']*100:.1f}%")
            st.metric("Latency", f"{result['latency_ms']:.0f}ms")
    except Exception as e:
        st.error(f"API Error: {e}")

# Analytics
st.markdown("---")
st.subheader("📊 Fraud Analytics")

# Merchant risk chart
merchant_fraud = df.groupby('merchant_category')['is_fraud'].mean() * 100
fig = go.Figure(go.Bar(x=merchant_fraud.index, y=merchant_fraud.values, marker_color='coral'))
fig.update_layout(title="Fraud Rate by Merchant", xaxis_title="Merchant", yaxis_title="Fraud Rate (%)")
st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption(f"Dashboard ready | API: {api_url}")