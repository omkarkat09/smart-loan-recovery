"""Application settings — paths and experiment config."""

from pathlib import Path

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path("C:/Users/dell/Documents/GitHub/smart-loan-recovery/data/raw")
PROCESSED_DIR = Path("C:/Users/dell/GitHub/smart-loan-recovery/data/processed")

RAW_DATA_PATH = str(DATA_DIR / "credit_risk_dataset.csv")
PROCESSED_DATA_PATH = str(PROCESSED_DIR / "processed_data.csv")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "file:///C:/Users/dell/GitHub/smart-loan-recovery/mlruns"

# ── Model ─────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
