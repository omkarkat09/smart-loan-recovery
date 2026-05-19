"""Phase 2 Step 3 — Assembled pipeline.

Chains: raw CSV → clean/encode (make_dataset) → feature engineer (build_features)
→ add customer_id → save as features.parquet.

All four stages run in sequence; output is a single parquet file for downstream
model training.
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import RAW_DATA_PATH
from src.data.make_dataset import load_raw, impute_nulls, cap_outliers, fix_dtypes, encode_categoricals, validate
from src.features.build_features import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("C:/Users/dell/GitHub/smart-loan-recovery/data/processed")
OUTPUT_PATH = OUTPUT_DIR / "features.parquet"
SEED = 42


def add_customer_id(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add a deterministic customer_id derived from a hash of row content.

    Since the raw dataset has no explicit customer_id, we create a stable
    one-from-content using the first 5 numeric columns + loan_grade + loan_status
    as a content fingerprint. This ensures the same row always gets the same ID
    across pipeline re-runs.
    """
    logger.info("Adding customer_id")
    import hashlib

    id_cols = [
        "person_age", "person_income", "person_emp_length",
        "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
    ]
    hash_input = df[id_cols].astype(str).agg("-".join, axis=1)
    customer_ids = hash_input.apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:16].upper()
    )
    df.insert(0, "customer_id", customer_ids)
    logger.info(f"  customer_id range: {df['customer_id'].iloc[0]} … {df['customer_id'].iloc[-1]}")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save to parquet with compression."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Saved → {path} | {len(df):,} rows × {df.shape[1]} cols | {path.stat().st_size / 1e6:.1f} MB")


def run_pipeline() -> pd.DataFrame:
    """Execute the full assembled pipeline and return the final DataFrame."""
    logger.info("=" * 60)
    logger.info("PHASE 2 STEP 3 — ASSEMBLED PIPELINE")
    logger.info("=" * 60)

    # ── Stage 1: Load raw ───────────────────────────────────────────────────
    logger.info("[1/5] Loading raw data")
    df = load_raw(RAW_DATA_PATH)
    logger.info(f"  Raw: {len(df):,} rows × {df.shape[1]} cols")

    # ── Stage 2: Clean & encode (from make_dataset) ─────────────────────────
    logger.info("[2/5] Cleaning & encoding")
    df = impute_nulls(df)
    df = cap_outliers(df)
    df = fix_dtypes(df)
    df = encode_categoricals(df)
    logger.info(f"  After clean/encode: {len(df):,} rows × {df.shape[1]} cols")

    # ── Stage 3: Feature engineering (from build_features) ──────────────────
    logger.info("[3/5] Building engineered features")
    df = build_features(df)
    logger.info(f"  After feature engineering: {len(df):,} rows × {df.shape[1]} cols")

    # ── Stage 4: Add customer_id ─────────────────────────────────────────────
    logger.info("[4/5] Merging on customer_id")
    df = add_customer_id(df)
    logger.info(f"  With customer_id: {len(df):,} rows × {df.shape[1]} cols")

    # ── Stage 5: Validate & save ─────────────────────────────────────────────
    logger.info("[5/5] Validating and saving as parquet")
    validate(df)
    save_parquet(df, OUTPUT_PATH)

    # ── Final summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — SUMMARY")
    logger.info(f"  Input : {RAW_DATA_PATH}")
    logger.info(f"  Output: {OUTPUT_PATH}")
    logger.info(f"  Rows  : {len(df):,}")
    logger.info(f"  Cols  : {df.shape[1]} ({df.shape[1] - 1} features + customer_id)")
    logger.info(f"  Default rate: {df['loan_status'].mean():.1%}")
    logger.info(f"  New features: {df.shape[1] - 18} engineered features added")
    logger.info(f"  All-null columns: {df.isnull().sum().sum()} (should be 0)")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run_pipeline()