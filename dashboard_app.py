"""
INTERACTIVE FRAUD DETECTION DASHBOARD
Real-time updates when settings change
Run: streamlit run dashboard/interactive_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1E88E5, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .fraud-box {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .legit-box {
        background-color: #00cc66;
        color: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🛡️ AI Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# LOAD DATA (WITHOUT CACHING FOR REAL-TIME UPDATES)
# ============================================

@st.cache_data(ttl=0)  # Disable caching for real-time updates
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/raw/credit_card_fraud_10k.csv')
        return df
    except:
        # Generate sample data
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'amount': np.random.exponential(100, n) + 10,
            'transaction_hour': np.random.randint(0, 24, n),
            'merchant_category': np.random.choice(['Electronics', 'Travel', 'Clothing', 'Food', 'Grocery'], n),
            'foreign_transaction': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'location_mismatch': np.random.choice([0, 1], n, p=[0.92, 0.08]),
            'device_trust_score': np.random.randint(0, 100, n),
            'velocity_last_24h': np.random.poisson(2, n),
            'cardholder_age': np.random.randint(18, 80, n),
            'is_fraud': np.random.choice([0, 1], n, p=[0.985, 0.015])
        })
        return df

# Load data (no caching)
df = load_data()

# ============================================
# SIDEBAR - Real-time Filters
# ============================================
st.sidebar.header("⚙️ Filters & Settings")

# Filter options
st.sidebar.subheader("📊 Data Filters")

# Merchant filter
merchants = st.sidebar.multiselect(
    "Merchant Category",
    options=df['merchant_category'].unique(),
    default=df['merchant_category'].unique().tolist()
)

# Amount range filter
amount_min, amount_max = st.sidebar.slider(
    "Transaction Amount ($)",
    min_value=float(df['amount'].min()),
    max_value=float(df['amount'].max()),
    value=(float(df['amount'].min()), float(df['amount'].max()))
)

# Device trust score filter
device_min, device_max = st.sidebar.slider(
    "Device Trust Score",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

# Hour range filter
hour_range = st.sidebar.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=(0, 23)
)

st.sidebar.markdown("---")

# Refresh button
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Apply filters to dataframe
filtered_df = df[
    (df['merchant_category'].isin(merchants)) &
    (df['amount'] >= amount_min) &
    (df['amount'] <= amount_max) &
    (df['device_trust_score'] >= device_min) &
    (df['device_trust_score'] <= device_max) &
    (df['transaction_hour'] >= hour_range[0]) &
    (df['transaction_hour'] <= hour_range[1])
]

# ============================================
# MAIN METRICS (Updates with filters)
# ============================================
st.subheader("📊 Live Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Transactions", f"{len(filtered_df):,}")
with col2:
    fraud_count = filtered_df['is_fraud'].sum()
    st.metric("Fraud Cases", f"{fraud_count:,}")
with col3:
    fraud_rate = (fraud_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
with col4:
    st.metric("Avg Amount", f"${filtered_df['amount'].mean():.2f}")
with col5:
    avg_fraud = filtered_df[filtered_df['is_fraud']==1]['amount'].mean() if fraud_count > 0 else 0
    st.metric("Avg Fraud Amount", f"${avg_fraud:.2f}")

st.markdown("---")

# ============================================
# PREDICTION FUNCTION (Rule-based for interactivity)
# ============================================

def calculate_risk_score(amount, hour, merchant, foreign, location_mismatch, device_score, velocity, age):
    """Calculate risk score based on transaction parameters"""
    risk_score = 0
    factors = []
    
    # Amount risk
    if amount > 1000:
        risk_score += 35
        factors.append("💰 Very high amount")
    elif amount > 500:
        risk_score += 25
        factors.append("💰 High amount")
    elif amount > 200:
        risk_score += 10
        factors.append("💰 Moderate amount")
    
    # Time risk
    if hour >= 22 or hour <= 5:
        risk_score += 20
        factors.append("🌙 Late night transaction")
    elif hour <= 8:
        risk_score += 10
        factors.append("🌅 Early morning transaction")
    
    # Location risk
    if foreign == 1:
        risk_score += 15
        factors.append("🌍 Foreign transaction")
    
    if location_mismatch == 1:
        risk_score += 20
        factors.append("📍 Location mismatch")
    
    # Device risk
    if device_score < 30:
        risk_score += 25
        factors.append("📱 Very low device trust")
    elif device_score < 50:
        risk_score += 15
        factors.append("📱 Low device trust")
    
    # Velocity risk
    if velocity > 5:
        risk_score += 15
        factors.append("⚡ High transaction velocity")
    elif velocity > 3:
        risk_score += 8
        factors.append("⚡ Moderate velocity")
    
    # Merchant risk
    if merchant in ['Electronics', 'Travel']:
        risk_score += 15
        factors.append(f"🏪 High-risk merchant: {merchant}")
    
    # Age risk
    if age < 25:
        risk_score += 8
        factors.append("👤 Young cardholder")
    elif age > 65:
        risk_score += 8
        factors.append("👤 Senior cardholder")
    
    return min(risk_score, 100), factors

# ============================================
# TEST TRANSACTION SECTION
# ============================================
st.subheader("🔍 Test Transaction")

col1, col2 = st.columns([1, 1])

with col1:
    with st.form("prediction_form"):
        st.markdown("### Transaction Details")
        
        transaction_id = st.text_input("Transaction ID", f"TX_{datetime.now().strftime('%H%M%S')}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            amount = st.number_input("Amount ($)", min_value=0.01, value=150.00, step=10.00, key="amount_input")
            hour = st.slider("Transaction Hour", 0, 23, 14, key="hour_input")
            merchant = st.selectbox("Merchant Category", ['Electronics', 'Travel', 'Clothing', 'Food', 'Grocery'], key="merchant_input")
        
        with col_b:
            foreign = st.selectbox("Foreign Transaction", [0, 1], format_func=lambda x: "Yes" if x else "No", key="foreign_input")
            location_mismatch = st.selectbox("Location Mismatch", [0, 1], format_func=lambda x: "Yes" if x else "No", key="location_input")
            device_score = st.slider("Device Trust Score", 0, 100, 75, key="device_input")
            velocity = st.number_input("Transactions in Last 24h", min_value=0, value=2, step=1, key="velocity_input")
            age = st.number_input("Cardholder Age", min_value=18, max_value=100, value=35, step=1, key="age_input")
        
        submitted = st.form_submit_button("🚨 Analyze Transaction", use_container_width=True)

with col2:
    st.markdown("### Risk Analysis")
    
    if submitted:
        # Calculate risk
        risk_score, risk_factors = calculate_risk_score(
            amount, hour, merchant, foreign, location_mismatch,
            device_score, velocity, age
        )
        
        is_fraud = risk_score >= 40
        fraud_prob = risk_score / 100
        
        # Display result
        if is_fraud:
            st.markdown(f"""
            <div class="fraud-box">
                <h2>⚠️ FRAUD ALERT!</h2>
                <p>Risk Score: {risk_score}/100</p>
                <p>Probability: {fraud_prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-box">
                <h2>✅ LEGITIMATE</h2>
                <p>Risk Score: {risk_score}/100</p>
                <p>Probability: {fraud_prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": "Risk Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if is_fraud else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Show risk factors
        if risk_factors:
            st.markdown("#### ⚠️ Risk Factors Detected")
            for factor in risk_factors:
                st.warning(factor)

st.markdown("---")

# ============================================
# LIVE ANALYTICS (Updates with filters)
# ============================================
st.subheader("📈 Live Fraud Analytics")

tab1, tab2, tab3, tab4 = st.tabs(["Hourly Pattern", "Merchant Risk", "Device Risk", "Amount Analysis"])

with tab1:
    # Hourly fraud rate - UPDATES WITH FILTERS
    hourly_fraud = filtered_df.groupby('transaction_hour')['is_fraud'].mean() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_fraud.index,
        y=hourly_fraud.values,
        mode='lines+markers',
        name='Fraud Rate',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='darkred')
    ))
    fig.add_hline(y=filtered_df['is_fraud'].mean()*100, line_dash="dash", line_color="blue", 
                 annotation_text="Average")
    fig.update_layout(
        title="Fraud Rate by Hour (Filtered View)",
        xaxis_title="Hour",
        yaxis_title="Fraud Rate (%)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if len(hourly_fraud) > 0:
        peak_hour = hourly_fraud.idxmax()
        st.info(f"🔴 Peak fraud hour: {peak_hour}:00 ({hourly_fraud.max():.1f}% fraud rate)")

with tab2:
    # Merchant fraud rate - UPDATES WITH FILTERS
    merchant_fraud = filtered_df.groupby('merchant_category')['is_fraud'].mean().sort_values() * 100
    
    colors = ['darkred' if x > filtered_df['is_fraud'].mean()*100 else 'green' for x in merchant_fraud.values]
    fig = go.Figure(go.Bar(
        x=merchant_fraud.values,
        y=merchant_fraud.index,
        orientation='h',
        marker_color=colors,
        text=merchant_fraud.values.round(1),
        textposition='outside'
    ))
    fig.update_layout(
        title="Fraud Rate by Merchant (Filtered View)",
        xaxis_title="Fraud Rate (%)",
        yaxis_title="Merchant Category",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Device score distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['is_fraud']==0]['device_trust_score'],
            name='Legitimate',
            opacity=0.7,
            marker_color='green'
        ))
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['is_fraud']==1]['device_trust_score'],
            name='Fraud',
            opacity=0.7,
            marker_color='red'
        ))
        fig.update_layout(
            title="Device Trust Score Distribution",
            xaxis_title="Device Trust Score",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Device score box plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['is_fraud']==0]['device_trust_score'],
            name='Legitimate',
            marker_color='green'
        ))
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['is_fraud']==1]['device_trust_score'],
            name='Fraud',
            marker_color='red'
        ))
        fig.update_layout(
            title="Device Score Comparison",
            yaxis_title="Device Trust Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['is_fraud']==0]['amount'],
            name='Legitimate',
            opacity=0.7,
            marker_color='green',
            nbinsx=50
        ))
        fig.add_trace(go.Histogram(
            x=filtered_df[filtered_df['is_fraud']==1]['amount'],
            name='Fraud',
            opacity=0.7,
            marker_color='red',
            nbinsx=50
        ))
        fig.update_layout(
            title="Transaction Amount Distribution",
            xaxis_title="Amount ($)",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Amount box plot
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['is_fraud']==0]['amount'],
            name='Legitimate',
            marker_color='green'
        ))
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['is_fraud']==1]['amount'],
            name='Fraud',
            marker_color='red'
        ))
        fig.update_layout(
            title="Amount Comparison",
            yaxis_title="Amount ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# SAMPLE TRANSACTIONS (Updates with filters)
# ============================================
st.subheader("📋 Sample Transactions (Filtered View)")

sample_df = filtered_df.head(20)[['amount', 'merchant_category', 'device_trust_score', 'transaction_hour', 'is_fraud']].copy()
sample_df['is_fraud'] = sample_df['is_fraud'].map({0: '✅ Legitimate', 1: '⚠️ Fraud'})
sample_df.columns = ['Amount', 'Merchant', 'Device Score', 'Hour', 'Status']
st.dataframe(sample_df, use_container_width=True)

# ============================================
# SUMMARY STATISTICS
# ============================================
st.markdown("---")
st.subheader("📊 Filter Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Filtered Transactions", f"{len(filtered_df):,} / {len(df):,}")
with col2:
    filtered_fraud_rate = (filtered_df['is_fraud'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    original_fraud_rate = (df['is_fraud'].sum() / len(df)) * 100
    st.metric("Fraud Rate", f"{filtered_fraud_rate:.2f}%", delta=f"{filtered_fraud_rate - original_fraud_rate:.2f}%")
with col3:
    st.metric("Active Filters", f"{len(merchants)} merchants | {hour_range[0]}-{hour_range[1]} hrs")

# Footer
st.markdown("---")
st.caption(f"""
    🛡️ Interactive Fraud Detection | Filters applied in real-time | 
    Last update: {datetime.now().strftime('%H:%M:%S')}
""")