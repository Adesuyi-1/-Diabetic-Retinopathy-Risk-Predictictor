"""
src/api.py
==========
FastAPI backend for Diabetic Retinopathy Risk Prediction.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
from pathlib import Path
from typing import List, Literal

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import config


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="Retinopathy Risk Prediction API",
    description="RandomForest-based diabetic retinopathy risk prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Model loading
# ============================================================
_model = None
_scaler = None
_feature_columns = None


def _load_model():
    global _model, _scaler, _feature_columns
    if _model is not None:
        return

    # Try final_random_forest first, then xgboost, then baseline
    candidates = [
        "final_random_forest.pkl",
        "baseline_random_forest.pkl",
        "xgboost_final.pkl",
    ]
    model_path = None
    for name in candidates:
        p = config.MODELS_DIR / name
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise RuntimeError(
            f"No model found in {config.MODELS_DIR}. "
            f"Tried: {candidates}"
        )
    _model = joblib.load(model_path)
    print(f"✅ Loaded model: {model_path.name}")

    # Preprocessor (or scaler)
    for name in ("preprocessor.pkl", "scaler.pkl"):
        p = config.MODELS_DIR / name
        if p.exists():
            _scaler = joblib.load(p)
            print(f"✅ Loaded preprocessor: {p.name}")
            break

    # Feature columns — from JSON or CSV
    cols_json = config.MODELS_DIR / "feature_columns.json"
    if cols_json.exists():
        with open(cols_json) as f:
            _feature_columns = json.load(f)
        print(f"✅ Loaded {len(_feature_columns)} feature names (JSON)")
    else:
        feat_csv = config.MODELS_DIR / "feature_importance.csv"
        if feat_csv.exists():
            feat_df = pd.read_csv(feat_csv)
            col = "feature" if "feature" in feat_df.columns else feat_df.columns[0]
            _feature_columns = feat_df[col].tolist()
            print(f"✅ Loaded {len(_feature_columns)} feature names (CSV)")


@app.on_event("startup")
async def startup():
    try:
        _load_model()
        print("✅ Startup complete")
    except Exception as e:
        print(f"⚠️ Startup model load failed: {e}")


# ============================================================
# Schemas
# ============================================================
class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    hba1c: float = Field(..., ge=4.0, le=20.0)
    bmi: float = Field(..., ge=10.0, le=60.0)
    diabetes_duration: int = Field(..., ge=0, le=60)
    systolic_bp: int = Field(..., ge=70, le=250)
    egfr: float = Field(..., ge=0, le=200)
    ldl: float = Field(..., ge=0.5, le=10.0)
    smoking: Literal["never", "former", "current"] = "never"
    insulin_use: Literal[0, 1] = 0
    ethnicity: str = "white"


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_percentage: float
    risk_tier: Literal["Low", "Medium", "High"]
    recommendation: str


class BatchRequest(BaseModel):
    patients: List[PatientData]


class BatchResponse(BaseModel):
    n_patients: int
    predictions: List[PredictionResponse]


# ============================================================
# Feature engineering
# NOTE: This is a GENERIC transformation. If /predict returns
#       feature-mismatch errors, paste your training script and
#       this will be rewritten to match exactly.
# ============================================================
def _to_dataframe(patient: PatientData) -> pd.DataFrame:
    row = patient.model_dump()

    row["smoking_current"] = 1 if row["smoking"] == "current" else 0
    row["smoking_former"] = 1 if row["smoking"] == "former" else 0
    row["ethnicity_south_asian"] = 1 if row["ethnicity"] == "south_asian" else 0
    row["ethnicity_black"] = 1 if row["ethnicity"] == "black" else 0
    row["ethnicity_other"] = 1 if row["ethnicity"] not in ("white", "south_asian", "black") else 0
    row["hba1c_x_duration"] = row["hba1c"] * row["diabetes_duration"]

    df = pd.DataFrame([row])
    df = df.drop(columns=["smoking", "ethnicity"], errors="ignore")

    if _feature_columns:
        df = df.reindex(columns=_feature_columns, fill_value=0)
    return df


def _predict_proba(df):
    if _scaler is not None:
        try:
            df = _scaler.transform(df)
        except Exception:
            pass
    return _model.predict_proba(df)[:, 1]


def _classify(prob: float):
    low = getattr(config, "RISK_THRESHOLD_LOW", 0.15)
    high = getattr(config, "RISK_THRESHOLD_HIGH", 0.45)
    if prob < low:
        return "Low", "Routine annual screening. Reinforce glycaemic and BP control."
    elif prob < high:
        return "Medium", "Consider 6-monthly screening. Optimise HbA1c, BP, lipids."
    else:
        return "High", "Urgent ophthalmology referral. Intensify glucose and BP control."


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "service": "Retinopathy Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "scaler_loaded": _scaler is not None,
        "feature_columns_loaded": _feature_columns is not None,
        "model_type": type(_model).__name__ if _model is not None else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = _to_dataframe(patient)
        prob = float(_predict_proba(df)[0])
        tier, rec = _classify(prob)
        return PredictionResponse(
            risk_probability=round(prob, 4),
            risk_percentage=round(prob * 100, 2),
            risk_tier=tier,
            recommendation=rec,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.patients:
        raise HTTPException(status_code=400, detail="Empty patient list")
    try:
        preds = []
        for p in req.patients:
            df = _to_dataframe(p)
            prob = float(_predict_proba(df)[0])
            tier, rec = _classify(prob)
            preds.append(PredictionResponse(
                risk_probability=round(prob, 4),
                risk_percentage=round(prob * 100, 2),
                risk_tier=tier,
                recommendation=rec,
            ))
        return BatchResponse(n_patients=len(preds), predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)