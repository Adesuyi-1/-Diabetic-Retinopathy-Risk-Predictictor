"""config.py — central config for the Retinopathy project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (MODELS_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
RISK_THRESHOLD_LOW = 0.15
RISK_THRESHOLD_HIGH = 0.45
