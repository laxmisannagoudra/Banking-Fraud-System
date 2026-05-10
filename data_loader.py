"""
STEP 1: Data Loader for Fraud Detection System
CSV file is located in src folder
"""

import pandas as pd
import os

print("=" * 60)
print("STEP 1: LOADING FRAUD DATA")
print("=" * 60)

# Load the data - CSV is in src folder
file_path = "src/credit_card_fraud_10k.csv"
print("\nLoading file: " + file_path)

# Check if file exists
if not os.path.exists(file_path):
    print("[ERROR] File not found: " + file_path)
    print("\nFiles in src folder:")
    for file in os.listdir('src'):
        print("  - " + file)
    exit(1)

# Read the CSV file
df = pd.read_csv(file_path)

print("\n[SUCCESS] Data loaded!")
print("Rows: " + str(len(df)))
print("Columns: " + str(len(df.columns)))
print("Column names: " + str(df.columns.tolist()))

# Check fraud count
fraud_count = df['is_fraud'].sum()
total = len(df)
fraud_rate = (fraud_count / total) * 100

print("\n" + "-" * 40)
print("FRAUD STATISTICS")
print("-" * 40)
print("Total transactions: " + str(total))
print("Fraud transactions: " + str(fraud_count))
print("Legit transactions: " + str(total - fraud_count))
print("Fraud rate: " + str(round(fraud_rate, 2)) + "%")

# Amount statistics
print("\n" + "-" * 40)
print("AMOUNT STATISTICS")
print("-" * 40)
print("Average amount: $" + str(round(df['amount'].mean(), 2)))
print("Max amount: $" + str(round(df['amount'].max(), 2)))
print("Min amount: $" + str(round(df['amount'].min(), 2)))

# Fraud vs Legit amounts
fraud_df = df[df['is_fraud'] == 1]
legit_df = df[df['is_fraud'] == 0]

print("\n" + "-" * 40)
print("FRAUD VS LEGITIMATE")
print("-" * 40)
print("Avg fraud amount: $" + str(round(fraud_df['amount'].mean(), 2)))
print("Avg legit amount: $" + str(round(legit_df['amount'].mean(), 2)))

if legit_df['amount'].mean() > 0:
    ratio = fraud_df['amount'].mean() / legit_df['amount'].mean()
    print("Fraud amounts are " + str(round(ratio, 1)) + "x higher")

# Device trust score comparison
print("\n" + "-" * 40)
print("DEVICE TRUST SCORE")
print("-" * 40)
print("Avg device score (fraud): " + str(round(fraud_df['device_trust_score'].mean(), 1)))
print("Avg device score (legit): " + str(round(legit_df['device_trust_score'].mean(), 1)))

# Foreign transaction comparison
print("\n" + "-" * 40)
print("FOREIGN TRANSACTIONS")
print("-" * 40)
print("Foreign tx fraud rate: " + str(round(fraud_df['foreign_transaction'].mean() * 100, 1)) + "%")
print("Foreign tx legit rate: " + str(round(legit_df['foreign_transaction'].mean() * 100, 1)) + "%")

# Location mismatch comparison
print("\n" + "-" * 40)
print("LOCATION MISMATCH")
print("-" * 40)
print("Location mismatch fraud rate: " + str(round(fraud_df['location_mismatch'].mean() * 100, 1)) + "%")
print("Location mismatch legit rate: " + str(round(legit_df['location_mismatch'].mean() * 100, 1)) + "%")

# Velocity comparison
print("\n" + "-" * 40)
print("TRANSACTION VELOCITY")
print("-" * 40)
print("Avg velocity (fraud): " + str(round(fraud_df['velocity_last_24h'].mean(), 1)))
print("Avg velocity (legit): " + str(round(legit_df['velocity_last_24h'].mean(), 1)))

# Create processed folder if needed
os.makedirs('data/processed', exist_ok=True)

# Save for next step
df.to_parquet('data/processed/fraud_data.parquet', index=False)
print("\n[SAVED] Data saved to: data/processed/fraud_data.parquet")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE!")
print("=" * 60)
print("\nNEXT: Run python src/feature_engineering.py")