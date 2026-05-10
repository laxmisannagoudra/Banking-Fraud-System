"""
STEP 2: Feature Engineering for Fraud Detection
Simplified with 2 key visualizations
Run: python src/feature_engineering.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load data from Step 1
print("\n[INFO] Loading data from Step 1...")
df = pd.read_parquet('data/processed/fraud_data.parquet')
print(f"[SUCCESS] Loaded {len(df):,} transactions")

print("\n[INFO] Creating engineered features...")

# ============================================
# 1. AMOUNT-BASED FEATURES
# ============================================
print("\n[1/8] Creating amount-based features...")

# Log transform amount (handles large values)
df['amount_log'] = np.log1p(df['amount'])

# Amount risk categories
df['amount_risk'] = 0
df.loc[df['amount'] > 200, 'amount_risk'] = 1
df.loc[df['amount'] > 500, 'amount_risk'] = 2
df.loc[df['amount'] > 1000, 'amount_risk'] = 3

# Amount unusual indicator
df['amount_unusual'] = np.where(df['amount'] > 500, 1, 0)

# ============================================
# 2. TIME-BASED FEATURES
# ============================================
print("[2/8] Creating time-based features...")

# Late night transactions (22:00 - 5:00)
df['is_late_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 5)).astype(int)

# Early morning transactions (5:00 - 8:00)
df['is_early_morning'] = ((df['transaction_hour'] >= 5) & (df['transaction_hour'] <= 8)).astype(int)

# Business hours (9:00 - 17:00)
df['is_business_hours'] = ((df['transaction_hour'] >= 9) & (df['transaction_hour'] <= 17)).astype(int)

# Cyclical encoding for hour
df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)

# ============================================
# 3. RISK SCORE FEATURES
# ============================================
print("[3/8] Creating risk score features...")

# Combined device risk (inverted: lower trust = higher risk)
df['device_ip_risk'] = (100 - df['device_trust_score'])

# Risk categories
df['device_risk_category'] = 'Low'
df.loc[df['device_trust_score'] < 30, 'device_risk_category'] = 'Critical'
df.loc[(df['device_trust_score'] >= 30) & (df['device_trust_score'] < 50), 'device_risk_category'] = 'High'
df.loc[(df['device_trust_score'] >= 50) & (df['device_trust_score'] < 70), 'device_risk_category'] = 'Medium'

# ============================================
# 4. VELOCITY FEATURES
# ============================================
print("[4/8] Creating velocity features...")

df['velocity_risk'] = 0
df.loc[df['velocity_last_24h'] > 2, 'velocity_risk'] = 1
df.loc[df['velocity_last_24h'] > 5, 'velocity_risk'] = 2
df.loc[df['velocity_last_24h'] > 10, 'velocity_risk'] = 3

df['is_high_velocity'] = (df['velocity_last_24h'] > 3).astype(int)

# ============================================
# 5. AGE-BASED FEATURES
# ============================================
print("[5/8] Creating age-based features...")

df['age_group'] = 'Adult'
df.loc[df['cardholder_age'] < 25, 'age_group'] = 'Young'
df.loc[df['cardholder_age'] > 60, 'age_group'] = 'Senior'

df['is_high_risk_age'] = ((df['cardholder_age'] < 25) | (df['cardholder_age'] > 65)).astype(int)

# ============================================
# 6. MERCHANT-BASED FEATURES
# ============================================
print("[6/8] Creating merchant-based features...")

merchant_risk_map = {
    'Electronics': 3,
    'Travel': 3,
    'Clothing': 2,
    'Food': 1,
    'Grocery': 1
}
df['merchant_risk'] = df['merchant_category'].map(merchant_risk_map).fillna(2)
df['is_high_risk_merchant'] = (df['merchant_risk'] >= 2).astype(int)

# ============================================
# 7. INTERACTION FEATURES
# ============================================
print("[7/8] Creating interaction features...")

df['risk_amount_interaction'] = df['device_ip_risk'] * df['amount_log']
df['night_foreign_interaction'] = df['is_late_night'] * df['foreign_transaction']
df['velocity_amount_interaction'] = df['velocity_last_24h'] * df['amount_log']
df['location_velocity_interaction'] = df['location_mismatch'] * df['velocity_last_24h']
df['risk_foreign_interaction'] = df['device_ip_risk'] * df['foreign_transaction']
df['age_amount_interaction'] = df['is_high_risk_age'] * df['amount_log']

# ============================================
# 8. ENCODE CATEGORICAL VARIABLES
# ============================================
print("[8/8] Encoding categorical variables...")

label_encoders = {}

le_merchant = LabelEncoder()
df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
label_encoders['merchant_category'] = le_merchant

le_device = LabelEncoder()
df['device_risk_category_encoded'] = le_device.fit_transform(df['device_risk_category'])
label_encoders['device_risk_category'] = le_device

le_age = LabelEncoder()
df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
label_encoders['age_group'] = le_age

print("\n[SUCCESS] Feature engineering complete!")

# ============================================
# FINAL FEATURE LIST
# ============================================
feature_columns = [
    'amount', 'amount_log', 'amount_risk', 'amount_unusual',
    'transaction_hour', 'hour_sin', 'hour_cos',
    'is_late_night', 'is_early_morning', 'is_business_hours',
    'foreign_transaction', 'location_mismatch',
    'device_trust_score', 'device_ip_risk', 'device_risk_category_encoded',
    'velocity_last_24h', 'velocity_risk', 'is_high_velocity',
    'cardholder_age', 'is_high_risk_age', 'age_group_encoded',
    'merchant_risk', 'is_high_risk_merchant', 'merchant_category_encoded',
    'risk_amount_interaction', 'night_foreign_interaction',
    'velocity_amount_interaction', 'location_velocity_interaction',
    'risk_foreign_interaction', 'age_amount_interaction'
]

print(f"\n[INFO] Total features created: {len(feature_columns)}")

# Create feature matrix
X = df[feature_columns]
y = df['is_fraud']

# Save feature-engineered data
df_features = pd.concat([X, y], axis=1)
df_features.to_parquet('data/processed/features.parquet', index=False)
print(f"\n[SAVED] Features saved to: data/processed/features.parquet")

# Save feature columns and encoders
joblib.dump(feature_columns, 'models/feature_columns.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')

# ============================================
# VISUALIZATION 1: Feature Correlation Heatmap
# ============================================
print("\n" + "=" * 70)
print("VISUALIZATION 1: Feature Correlation Heatmap")
print("=" * 70)

plt.figure(figsize=(12, 10))

# Get top 15 features correlated with fraud
corr_matrix = df_features.corr()
target_corr = corr_matrix['is_fraud'].abs().sort_values(ascending=False)
top_features = target_corr.head(15).index
corr_top = df_features[top_features].corr()

# Create heatmap
sns.heatmap(corr_top, annot=True, fmt='.2f', cmap='RdYlGn_r', square=True, 
            annot_kws={'size': 8})
plt.title('Top 15 Feature Correlations with Fraud Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/feature_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("[SAVED] reports/feature_correlation_heatmap.png")

# Print correlation values
print("\nTop 10 Features Correlated with Fraud:")
correlations = df_features.corr()['is_fraud'].abs().sort_values(ascending=False)
for i, (feature, corr) in enumerate(correlations.head(11).items(), 1):
    if feature != 'is_fraud':
        bar = '=' * int(corr * 40)
        print(f"{i-1:2d}. {feature:30s} {bar:40s} {corr:.4f}")

# ============================================
# VISUALIZATION 2: Fraud Patterns Dashboard
# ============================================
print("\n" + "=" * 70)
print("VISUALIZATION 2: Fraud Patterns Dashboard")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2.1: Amount Distribution (Fraud vs Legit)
ax1 = axes[0, 0]
fraud_amounts = df[df['is_fraud'] == 1]['amount']
legit_amounts = df[df['is_fraud'] == 0]['amount']
ax1.hist([legit_amounts, fraud_amounts], bins=50, alpha=0.7, 
         label=['Legitimate', 'Fraud'], color=['green', 'red'])
ax1.set_xlabel('Transaction Amount ($)')
ax1.set_ylabel('Frequency')
ax1.set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.set_yscale('log')

# 2.2: Device Trust Score Boxplot
ax2 = axes[0, 1]
df.boxplot(column='device_trust_score', by='is_fraud', ax=ax2)
ax2.set_title('Device Trust Score by Transaction Type', fontsize=12, fontweight='bold')
ax2.set_xlabel('Transaction Type (0=Legitimate, 1=Fraud)')
ax2.set_ylabel('Device Trust Score')

# 2.3: Hourly Fraud Rate
ax3 = axes[1, 0]
hourly_fraud = df.groupby('transaction_hour')['is_fraud'].mean() * 100
ax3.plot(hourly_fraud.index, hourly_fraud.values, 'ro-', linewidth=2, markersize=8)
ax3.fill_between(hourly_fraud.index, 0, hourly_fraud.values, alpha=0.3, color='red')
ax3.set_xlabel('Hour of Day')
ax3.set_ylabel('Fraud Rate (%)')
ax3.set_title('Fraud Rate by Transaction Hour', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=df['is_fraud'].mean()*100, color='blue', linestyle='--', label='Average Fraud Rate')
ax3.legend()

# 2.4: Merchant Category Fraud Rate
ax4 = axes[1, 1]
merchant_fraud = df.groupby('merchant_category')['is_fraud'].mean().sort_values() * 100
colors = ['red' if x > df['is_fraud'].mean()*100 else 'green' for x in merchant_fraud.values]
merchant_fraud.plot(kind='barh', ax=ax4, color=colors)
ax4.set_xlabel('Fraud Rate (%)')
ax4.set_title('Fraud Rate by Merchant Category', fontsize=12, fontweight='bold')
ax4.axvline(x=df['is_fraud'].mean()*100, color='blue', linestyle='--', linewidth=2, label='Average')
ax4.legend()

plt.suptitle('Fraud Detection Pattern Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/fraud_patterns_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("[SAVED] reports/fraud_patterns_dashboard.png")

print("\n" + "=" * 70)
print("STEP 2 COMPLETE! Ready for Model Training")
print("=" * 70)
print("\n[NEXT] Run: python src/train_model.py")