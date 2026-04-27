"""Load raw CSV, clean, impute, cap outliers, encode categoricals, save processed dataset.

Phase 2 — Step 1: Clean the data
- Impute nulls: median for numeric, mode for categorical
- Fix dtypes: downcast integers, normalise strings
- Cap outliers: apply domain-informed bounds per column
- Encode categoricals: ordinal + binary + one-hot
- Save to data/processed/
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Domain-informed outlier caps ──────────────────────────────────────────────
OUTLIER_CAPS = {
    "person_age":                (18, 100),    # realistic working age range
    "person_income":             (5_000, 1_000_000),   # $5k–$1M
    "person_emp_length":         (0, 40),      # 0–40 years
    "loan_amnt":                 (500, 50_000), # $500–$50k
    "loan_int_rate":             (3.0, 25.0),  # 3–25%
    "loan_percent_income":       (0.5, 50.0),  # 0.5–50% of income
    "cb_person_cred_hist_length": (1, 40),      # 1–40 years
}

# ── Null-imputation strategy ──────────────────────────────────────────────────
# public so tests can import them
IMPUTE_NUMERIC_MEDIAN = [
    "person_emp_length",
    "loan_int_rate",
]

IMPUTE_CATEGORICAL_MODE = [
    "person_home_ownership",
    "loan_intent",
]


# ── Pipeline functions ───────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load the raw credit risk CSV."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns — cols: {list(df.columns)}")
    return df


def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    logger.info("Imputing nulls")

    for col in IMPUTE_NUMERIC_MEDIAN:
        if col in df.columns:
            median_val = df[col].median()
            null_count = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: imputed {null_count:,} nulls with median={median_val:.2f}")

    for col in IMPUTE_CATEGORICAL_MODE:
        if col in df.columns:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN"
            null_count = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  {col}: imputed {null_count:,} nulls with mode='{mode_val}'")

    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values outside domain-informed bounds to the bounds."""
    logger.info("Capping outliers")
    for col, (low, high) in OUTLIER_CAPS.items():
        if col not in df.columns:
            continue
        before = df[col].copy()
        df[col] = df[col].clip(lower=low, upper=high)
        clipped = (before != df[col]).sum()
        if clipped > 0:
            logger.info(f"  {col}: clipped {clipped:,} values to [{low}, {high}]")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerics and normalise string columns."""
    logger.info("Fixing dtypes")

    # Normalise string columns: strip whitespace, title-case
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip().str.title()

    # Downcast int columns
    int_cols = df.select_dtypes(include="int").columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Downcast float columns
    float_cols = df.select_dtypes(include="float").columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    logger.info(f"Dtypes after fix: {dict(df.dtypes)}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns: ordinal grade, binary default flag, one-hot rest."""
    logger.info("Encoding categoricals")

    # Binary encode: cb_person_default_on_file  (Y=1, N=0)
    df["cb_person_default_on_file"] = (
        df["cb_person_default_on_file"].str.strip().str.lower() == "y"
    ).astype("int8")

    # Ordinal encode: loan_grade  (A=1 … G=7)
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["loan_grade"] = (
        df["loan_grade"].str.strip().str.upper().map(grade_map).astype("int8")
    )

    # One-hot encode: person_home_ownership, loan_intent
    one_hot_cols = [
        c for c in ["person_home_ownership", "loan_intent"] if c in df.columns
    ]
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype="int8")

    logger.info(f"After encoding: {df.shape[1]} columns — {list(df.columns)}")
    return df


def validate(df: pd.DataFrame) -> None:
    """Sanity-check the processed DataFrame."""
    logger.info("Running validation checks")
    assert df.isnull().sum().sum() == 0, "Unexpected null values remain!"
    assert df.select_dtypes(include="float").columns.notna().all(), "Float cols should be non-null"
    assert df["loan_status"].isin([0, 1]).all(), "loan_status must be binary!"
    assert 0 < df["loan_status"].mean() < 1, "loan_status has no variance!"
    logger.info(
        f"Validation passed — {len(df):,} rows × {df.shape[1]} cols | "
        f"Default rate: {df['loan_status'].mean():.3f}"
    )


def main():
    """Run the full Phase 2 Step 1 pipeline."""
    raw_path  = RAW_DATA_PATH
    out_path  = PROCESSED_DATA_PATH

    df = load_raw(raw_path)
    df = impute_nulls(df)
    df = cap_outliers(df)
    df = fix_dtypes(df)
    df = encode_categoricals(df)
    validate(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        f"Saved → {out_path} | {len(df):,} rows × {df.shape[1]} cols | "
        f"Defaults: {df['loan_status'].sum():,} ({df['loan_status'].mean():.1%})"
    )


if __name__ == "__main__":
    main()