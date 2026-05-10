"""
STEP 3: Train MULTIPLE Fraud Detection Models and Compare
Models: Logistic Regression, Random Forest, XGBoost
Run: python src/train_all_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve,
                             accuracy_score, f1_score)
import xgboost as xgb
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 3: TRAINING MULTIPLE FRAUD DETECTION MODELS")
print("=" * 70)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load features
print("\n[INFO] Loading feature data...")

df = pd.read_parquet('data/processed/features.parquet')
feature_cols = joblib.load('models/feature_columns.pkl')

X = df[feature_cols]
y = df['is_fraud']

print(f"[SUCCESS] Loaded {len(X):,} transactions")
print(f"   Features: {len(feature_cols)}")
print(f"   Fraud rate: {y.mean()*100:.2f}%")
print(f"   Fraud cases: {y.sum():,}")
print(f"   Legit cases: {(y==0).sum():,}")

# ============================================
# SPLIT DATA
# ============================================
print("\n" + "-" * 60)
print("DATA SPLIT")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} ({y_train.mean()*100:.2f}% fraud)")
print(f"Test set:     {len(X_test):,} ({y_test.mean()*100:.2f}% fraud)")

# ============================================
# SCALE FEATURES
# ============================================
print("\n" + "-" * 60)
print("FEATURE SCALING")
print("-" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[OK] Features scaled")

# Calculate class weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

# ============================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================
print("\n" + "=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

print("\nTraining Logistic Regression...")
start_time = time.time()

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=1.0
)
lr_model.fit(X_train_scaled, y_train)

lr_train_time = time.time() - start_time
print(f"Training time: {lr_train_time:.2f} seconds")

# Predictions
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, lr_proba)
if len(thresholds) > 0:
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    lr_optimal_threshold = thresholds[np.argmax(f1_scores)]
    lr_pred_opt = (lr_proba >= lr_optimal_threshold).astype(int)
else:
    lr_optimal_threshold = 0.5
    lr_pred_opt = lr_pred

# Metrics
lr_tn, lr_fp, lr_fn, lr_tp = confusion_matrix(y_test, lr_pred_opt).ravel()
lr_recall = lr_tp / (lr_tp + lr_fn) if (lr_tp + lr_fn) > 0 else 0
lr_precision = lr_tp / (lr_tp + lr_fp) if (lr_tp + lr_fp) > 0 else 0
lr_fpr = lr_fp / (lr_fp + lr_tn) if (lr_fp + lr_tn) > 0 else 0
lr_auc = roc_auc_score(y_test, lr_proba)
lr_f1 = f1_score(y_test, lr_pred_opt)

print(f"\nResults:")
print(f"   Optimal Threshold: {lr_optimal_threshold:.3f}")
print(f"   Fraud Recall: {lr_recall*100:.2f}%")
print(f"   Fraud Precision: {lr_precision*100:.2f}%")
print(f"   False Positive Rate: {lr_fpr*100:.2f}%")
print(f"   F1 Score: {lr_f1:.4f}")
print(f"   ROC-AUC: {lr_auc:.4f}")

# ============================================
# MODEL 2: RANDOM FOREST
# ============================================
print("\n" + "=" * 60)
print("MODEL 2: RANDOM FOREST")
print("=" * 60)

print("\nTraining Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

rf_train_time = time.time() - start_time
print(f"Training time: {rf_train_time:.2f} seconds")

# Predictions
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_proba)
if len(thresholds) > 0:
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    rf_optimal_threshold = thresholds[np.argmax(f1_scores)]
    rf_pred_opt = (rf_proba >= rf_optimal_threshold).astype(int)
else:
    rf_optimal_threshold = 0.5
    rf_pred_opt = rf_pred

# Metrics
rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(y_test, rf_pred_opt).ravel()
rf_recall = rf_tp / (rf_tp + rf_fn) if (rf_tp + rf_fn) > 0 else 0
rf_precision = rf_tp / (rf_tp + rf_fp) if (rf_tp + rf_fp) > 0 else 0
rf_fpr = rf_fp / (rf_fp + rf_tn) if (rf_fp + rf_tn) > 0 else 0
rf_auc = roc_auc_score(y_test, rf_proba)
rf_f1 = f1_score(y_test, rf_pred_opt)

print(f"\nResults:")
print(f"   Optimal Threshold: {rf_optimal_threshold:.3f}")
print(f"   Fraud Recall: {rf_recall*100:.2f}%")
print(f"   Fraud Precision: {rf_precision*100:.2f}%")
print(f"   False Positive Rate: {rf_fpr*100:.2f}%")
print(f"   F1 Score: {rf_f1:.4f}")
print(f"   ROC-AUC: {rf_auc:.4f}")

# ============================================
# MODEL 3: XGBOOST
# ============================================
print("\n" + "=" * 60)
print("MODEL 3: XGBOOST")
print("=" * 60)

print("\nTraining XGBoost...")
start_time = time.time()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

xgb_train_time = time.time() - start_time
print(f"Training time: {xgb_train_time:.2f} seconds")

# Predictions
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, xgb_proba)
if len(thresholds) > 0:
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    xgb_optimal_threshold = thresholds[np.argmax(f1_scores)]
    xgb_pred_opt = (xgb_proba >= xgb_optimal_threshold).astype(int)
else:
    xgb_optimal_threshold = 0.5
    xgb_pred_opt = xgb_model.predict(X_test_scaled)

# Metrics
xgb_tn, xgb_fp, xgb_fn, xgb_tp = confusion_matrix(y_test, xgb_pred_opt).ravel()
xgb_recall = xgb_tp / (xgb_tp + xgb_fn) if (xgb_tp + xgb_fn) > 0 else 0
xgb_precision = xgb_tp / (xgb_tp + xgb_fp) if (xgb_tp + xgb_fp) > 0 else 0
xgb_fpr = xgb_fp / (xgb_fp + xgb_tn) if (xgb_fp + xgb_tn) > 0 else 0
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_f1 = f1_score(y_test, xgb_pred_opt)

print(f"\nResults:")
print(f"   Optimal Threshold: {xgb_optimal_threshold:.3f}")
print(f"   Fraud Recall: {xgb_recall*100:.2f}%")
print(f"   Fraud Precision: {xgb_precision*100:.2f}%")
print(f"   False Positive Rate: {xgb_fpr*100:.2f}%")
print(f"   F1 Score: {xgb_f1:.4f}")
print(f"   ROC-AUC: {xgb_auc:.4f}")

# ============================================
# COMPARE ALL MODELS
# ============================================
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Training Time (s)': [lr_train_time, rf_train_time, xgb_train_time],
    'Fraud Recall (%)': [lr_recall*100, rf_recall*100, xgb_recall*100],
    'Precision (%)': [lr_precision*100, rf_precision*100, xgb_precision*100],
    'FPR (%)': [lr_fpr*100, rf_fpr*100, xgb_fpr*100],
    'F1 Score': [lr_f1, rf_f1, xgb_f1],
    'ROC-AUC': [lr_auc, rf_auc, xgb_auc]
})

print("\n" + comparison_df.to_string(index=False))

# Find best model for each metric
best_recall = comparison_df.loc[comparison_df['Fraud Recall (%)'].idxmax(), 'Model']
best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
best_auc = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']

print("\n" + "-" * 70)
print("BEST MODELS BY METRIC")
print("-" * 70)
print(f"Best Fraud Recall: {best_recall}")
print(f"Best F1 Score: {best_f1}")
print(f"Best ROC-AUC: {best_auc}")

# Recommend best overall model (based on F1 score)
best_model_name = best_f1
print(f"\n[RECOMMENDATION] Best overall model: {best_model_name}")

# ============================================
# SAVE BEST MODEL
# ============================================
print("\n" + "-" * 60)
print("SAVING BEST MODEL AND ARTIFACTS")
print("-" * 60)

# Save the best model
if best_model_name == 'Logistic Regression':
    final_model = lr_model
    final_threshold = lr_optimal_threshold
elif best_model_name == 'Random Forest':
    final_model = rf_model
    final_threshold = rf_optimal_threshold
else:
    final_model = xgb_model
    final_threshold = xgb_optimal_threshold

joblib.dump(final_model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(final_threshold, 'models/threshold.pkl')
joblib.dump(feature_cols, 'models/feature_columns.pkl')

print(f"[SAVED] Best model ({best_model_name}) to models/fraud_model.pkl")
print("[SAVED] scaler.pkl")
print("[SAVED] threshold.pkl")
print("[SAVED] feature_columns.pkl")

# Save all models for reference
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(xgb_model, 'models/xgboost_model.pkl')

print("[SAVED] All individual models for reference")

# ============================================
# FEATURE IMPORTANCE (from Random Forest)
# ============================================
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE (Random Forest)")
print("-" * 60)

rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in rf_importance.head(10).iterrows():
    bar = '=' * int(row['importance'] * 50)
    print(f"{row['feature']:35s} {bar:40s} {row['importance']:.4f}")

rf_importance.to_csv('reports/feature_importance_rf.csv', index=False)

# ============================================
# SAVE COMPARISON RESULTS
# ============================================
comparison_df.to_csv('reports/model_comparison.csv', index=False)
print("\n[SAVED] Model comparison to reports/model_comparison.csv")

# Save test predictions from best model
if best_model_name == 'Logistic Regression':
    best_proba = lr_proba
    best_pred = lr_pred_opt
elif best_model_name == 'Random Forest':
    best_proba = rf_proba
    best_pred = rf_pred_opt
else:
    best_proba = xgb_proba
    best_pred = xgb_pred_opt

results = pd.DataFrame({
    'actual': y_test.values,
    'predicted': best_pred,
    'probability': best_proba,
    'model_used': best_model_name
})
results.to_csv('reports/test_predictions.csv', index=False)
print("[SAVED] Test predictions to reports/test_predictions.csv")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("STEP 3 COMPLETE!")
print("=" * 70)

print(f"\nFINAL MODEL SUMMARY:")
print(f"   Best Model: {best_model_name}")
print(f"   Fraud Recall: {comparison_df.loc[comparison_df['Model']==best_model_name, 'Fraud Recall (%)'].values[0]:.1f}%")
print(f"   False Positive Rate: {comparison_df.loc[comparison_df['Model']==best_model_name, 'FPR (%)'].values[0]:.1f}%")
print(f"   F1 Score: {comparison_df.loc[comparison_df['Model']==best_model_name, 'F1 Score'].values[0]:.4f}")
print(f"   ROC-AUC: {comparison_df.loc[comparison_df['Model']==best_model_name, 'ROC-AUC'].values[0]:.4f}")

print("\n[NEXT] Step 4: Deploy API")
print("   Command: python src/fraud_api.py")