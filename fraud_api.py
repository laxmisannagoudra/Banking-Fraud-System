"""
STEP 4: Fraud Detection API - Real-time prediction service (Fixed)
Run: python src/fraud_api.py
Then open: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time
import os

print("=" * 70)
print("STEP 4: FRAUD DETECTION API")
print("=" * 70)

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using XGBoost",
    version="2.0.0"
)

# Global variables for models
model = None
scaler = None
threshold = None
feature_cols = None
label_encoders = None

# ============================================
# DATA MODELS (Request/Response Schemas)
# ============================================

class Transaction(BaseModel):
    """Single transaction for fraud prediction"""
    transaction_id: str
    amount: float = Field(..., gt=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    merchant_category: str
    foreign_transaction: int = Field(..., ge=0, le=1)
    location_mismatch: int = Field(..., ge=0, le=1)
    device_trust_score: int = Field(..., ge=0, le=100)
    velocity_last_24h: int = Field(..., ge=0)
    cardholder_age: int = Field(..., ge=18, le=100)

class FraudResponse(BaseModel):
    """Fraud prediction response"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    latency_ms: float
    timestamp: str

class BatchTransaction(BaseModel):
    """Batch transaction request"""
    transactions: List[Transaction]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str

# ============================================
# LOAD MODELS ON STARTUP
# ============================================

@app.on_event("startup")
async def load_models():
    """Load pre-trained models and artifacts"""
    global model, scaler, threshold, feature_cols, label_encoders
    
    print("\n[INFO] Loading fraud detection models...")
    
    try:
        model = joblib.load('models/fraud_model.pkl')
        print("[OK] Loaded fraud_model.pkl")
    except Exception as e:
        print(f"[ERROR] Could not load fraud_model.pkl: {e}")
        return
    
    try:
        scaler = joblib.load('models/scaler.pkl')
        print("[OK] Loaded scaler.pkl")
    except Exception as e:
        print(f"[ERROR] Could not load scaler.pkl: {e}")
        return
    
    try:
        threshold = float(joblib.load('models/threshold.pkl'))
        print(f"[OK] Loaded threshold: {threshold:.3f}")
    except Exception as e:
        print(f"[WARNING] Using default threshold 0.5")
        threshold = 0.5
    
    try:
        feature_cols = joblib.load('models/feature_columns.pkl')
        print(f"[OK] Loaded {len(feature_cols)} feature columns")
    except Exception as e:
        print(f"[ERROR] Could not load feature_columns.pkl: {e}")
        return
    
    try:
        label_encoders = joblib.load('models/label_encoders.pkl')
        print("[OK] Loaded label encoders")
    except Exception as e:
        print(f"[WARNING] Could not load label encoders: {e}")
        label_encoders = None
    
    print("\n[SUCCESS] All models loaded! API is ready.")
    print("=" * 70)

# ============================================
# FEATURE EXTRACTION FUNCTION
# ============================================

def extract_features(transaction: Transaction) -> np.ndarray:
    """Extract all 30 features from transaction data"""
    
    # Create feature dictionary
    features = {}
    
    # Amount-based features
    features['amount'] = float(transaction.amount)
    features['amount_log'] = float(np.log1p(transaction.amount))
    
    if transaction.amount > 1000:
        features['amount_risk'] = 3
    elif transaction.amount > 500:
        features['amount_risk'] = 2
    elif transaction.amount > 200:
        features['amount_risk'] = 1
    else:
        features['amount_risk'] = 0
    
    features['amount_unusual'] = 1 if transaction.amount > 500 else 0
    
    # Time-based features
    features['transaction_hour'] = transaction.transaction_hour
    features['hour_sin'] = float(np.sin(2 * np.pi * transaction.transaction_hour / 24))
    features['hour_cos'] = float(np.cos(2 * np.pi * transaction.transaction_hour / 24))
    features['is_late_night'] = 1 if (transaction.transaction_hour >= 22 or transaction.transaction_hour <= 5) else 0
    features['is_early_morning'] = 1 if (5 <= transaction.transaction_hour <= 8) else 0
    features['is_business_hours'] = 1 if (9 <= transaction.transaction_hour <= 17) else 0
    
    # Risk-based features
    features['foreign_transaction'] = transaction.foreign_transaction
    features['location_mismatch'] = transaction.location_mismatch
    features['device_trust_score'] = transaction.device_trust_score
    features['device_ip_risk'] = float(100 - transaction.device_trust_score)
    
    # Device risk category (simplified)
    if transaction.device_trust_score < 30:
        device_category = 'Critical'
    elif transaction.device_trust_score < 50:
        device_category = 'High'
    elif transaction.device_trust_score < 70:
        device_category = 'Medium'
    else:
        device_category = 'Low'
    
    # Simple encoding for device risk (if no label encoder)
    device_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}
    features['device_risk_category_encoded'] = device_map.get(device_category, 0)
    
    # Velocity features
    features['velocity_last_24h'] = transaction.velocity_last_24h
    
    if transaction.velocity_last_24h > 10:
        features['velocity_risk'] = 3
    elif transaction.velocity_last_24h > 5:
        features['velocity_risk'] = 2
    elif transaction.velocity_last_24h > 2:
        features['velocity_risk'] = 1
    else:
        features['velocity_risk'] = 0
    
    features['is_high_velocity'] = 1 if transaction.velocity_last_24h > 3 else 0
    
    # Age features
    features['cardholder_age'] = transaction.cardholder_age
    features['is_high_risk_age'] = 1 if (transaction.cardholder_age < 25 or transaction.cardholder_age > 65) else 0
    
    # Age group (simplified)
    if transaction.cardholder_age < 25:
        age_group = 'Young'
    elif transaction.cardholder_age < 40:
        age_group = 'Adult'
    elif transaction.cardholder_age < 60:
        age_group = 'Middle'
    else:
        age_group = 'Senior'
    
    age_map = {'Young': 0, 'Adult': 1, 'Middle': 2, 'Senior': 3}
    features['age_group_encoded'] = age_map.get(age_group, 1)
    
    # Merchant features
    merchant_risk_map = {'Electronics': 3, 'Travel': 3, 'Clothing': 2, 'Food': 1, 'Grocery': 1}
    features['merchant_risk'] = merchant_risk_map.get(transaction.merchant_category, 2)
    features['is_high_risk_merchant'] = 1 if features['merchant_risk'] >= 2 else 0
    
    # Simple merchant encoding
    merchant_map = {'Electronics': 0, 'Travel': 1, 'Clothing': 2, 'Food': 3, 'Grocery': 4}
    features['merchant_category_encoded'] = merchant_map.get(transaction.merchant_category, 0)
    
    # Interaction features
    features['risk_amount_interaction'] = float(features['device_ip_risk'] * features['amount_log'])
    features['night_foreign_interaction'] = float(features['is_late_night'] * transaction.foreign_transaction)
    features['velocity_amount_interaction'] = float(transaction.velocity_last_24h * features['amount_log'])
    features['location_velocity_interaction'] = float(transaction.location_mismatch * transaction.velocity_last_24h)
    features['risk_foreign_interaction'] = float(features['device_ip_risk'] * transaction.foreign_transaction)
    features['age_amount_interaction'] = float(features['is_high_risk_age'] * features['amount_log'])
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Ensure all feature columns exist
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]
    
    return df.values.astype(np.float32)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        message="Fraud Detection API is running"
    )

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": float(threshold) if threshold else 0.5
    }

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: Transaction):
    """Predict if a single transaction is fraudulent"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract features
        features = extract_features(transaction)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get prediction
        fraud_prob = float(model.predict_proba(features_scaled)[0, 1])
        is_fraud = fraud_prob >= threshold
        risk_score = float(fraud_prob * 100)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return FraudResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=bool(is_fraud),
            fraud_probability=round(fraud_prob, 4),
            risk_score=round(risk_score, 2),
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchTransaction):
    """Predict fraud for multiple transactions"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    fraud_count = 0
    
    for transaction in request.transactions:
        try:
            features = extract_features(transaction)
            features_scaled = scaler.transform(features)
            fraud_prob = float(model.predict_proba(features_scaled)[0, 1])
            is_fraud = fraud_prob >= threshold
            
            if is_fraud:
                fraud_count += 1
            
            results.append({
                "transaction_id": transaction.transaction_id,
                "is_fraud": bool(is_fraud),
                "fraud_probability": round(fraud_prob, 4),
                "risk_score": round(fraud_prob * 100, 2)
            })
        except Exception as e:
            results.append({
                "transaction_id": transaction.transaction_id,
                "error": str(e)
            })
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "total_transactions": len(request.transactions),
        "fraud_count": fraud_count,
        "results": results,
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": "XGBoost",
        "features_count": len(feature_cols) if feature_cols else 0,
        "threshold": float(threshold) if threshold else 0.5,
        "status": "loaded"
    }

# ============================================
# RUN THE API
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("STARTING FRAUD DETECTION API")
    print("=" * 70)
    print("\n[INFO] API Documentation: http://localhost:8000/docs")
    print("[INFO] Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)