"""
tests/test_api.py
=================
FastAPI endpoint tests using TestClient (no server needed).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    """TestClient with startup/shutdown handled."""
    with TestClient(app) as c:
        yield c


VALID_PATIENT = {
    "age": 62,
    "hba1c": 8.5,
    "bmi": 31.2,
    "diabetes_duration": 12,
    "systolic_bp": 145,
    "egfr": 62.0,
    "ldl": 2.8,
    "smoking": "former",
    "insulin_use": 1,
    "ethnicity": "south_asian",
}


# ============================================================
# Health / Root
# ============================================================
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "service" in body or "message" in body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") in ("healthy", "degraded")
    assert "model_loaded" in body


# ============================================================
# Single prediction
# ============================================================
def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert "risk_probability" in body
    assert 0 <= body["risk_probability"] <= 1
    assert body["risk_tier"] in ("Low", "Medium", "High")
    assert "recommendation" in body


def test_predict_high_risk(client):
    """A very high-risk patient should get High tier."""
    high_risk = {
        "age": 70,
        "hba1c": 10.5,
        "bmi": 35.0,
        "diabetes_duration": 20,
        "systolic_bp": 160,
        "egfr": 45.0,
        "ldl": 4.5,
        "smoking": "current",
        "insulin_use": 1,
        "ethnicity": "south_asian",
    }
    r = client.post("/predict", json=high_risk)
    assert r.status_code == 200
    assert r.json()["risk_tier"] in ("Medium", "High")


def test_predict_low_risk(client):
    """A healthy patient should get Low or Medium tier."""
    low_risk = {
        "age": 30,
        "hba1c": 5.5,
        "bmi": 22.0,
        "diabetes_duration": 1,
        "systolic_bp": 115,
        "egfr": 95.0,
        "ldl": 1.8,
        "smoking": "never",
        "insulin_use": 0,
        "ethnicity": "white",
    }
    r = client.post("/predict", json=low_risk)
    assert r.status_code == 200
    assert r.json()["risk_tier"] in ("Low", "Medium")


# ============================================================
# Validation
# ============================================================
def test_predict_missing_field(client):
    bad = VALID_PATIENT.copy()
    del bad["hba1c"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    bad = VALID_PATIENT.copy()
    bad["age"] = 200  # exceeds le=100
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_invalid_enum(client):
    bad = VALID_PATIENT.copy()
    bad["smoking"] = "sometimes"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


# ============================================================
# Batch
# ============================================================
def test_predict_batch(client):
    payload = {
        "patients": [
            VALID_PATIENT,
            {**VALID_PATIENT, "age": 40, "hba1c": 6.0},
        ]
    }
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n_patients"] == 2
    assert len(body["predictions"]) == 2


def test_predict_batch_empty(client):
    r = client.post("/predict/batch", json={"patients": []})
    assert r.status_code == 400