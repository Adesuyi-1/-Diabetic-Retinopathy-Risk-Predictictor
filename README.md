# 🩺 Diabetic Retinopathy Risk Prediction

A machine learning system for predicting diabetic retinopathy risk in type 2 diabetes patients using routine clinical data. Includes XGBoost/Random Forest models with SHAP explainability, a clinician-facing Streamlit interface, and a FastAPI backend for programmatic access.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

Diabetic retinopathy is the leading cause of preventable blindness in working-age adults. Early detection is critical — but screening capacity is limited. This project builds a **risk prediction model** that uses routine clinical data (HbA1c, diabetes duration, kidney function, etc.) to **prioritise high-risk patients for retinal screening**.

The system includes:

- **Machine learning model** (Random Forest / XGBoost) trained on ~5,000 synthetic patient records
- **SHAP explainability** — every prediction comes with feature-level reasoning
- **Calibration analysis** — clinically valid risk probabilities, not just ranking
- **Streamlit web interface** — clinician-friendly risk calculator
- **FastAPI REST API** — programmatic access for integration
- **Docker support** — one-command deployment

---

## 🏗️ Architecture
