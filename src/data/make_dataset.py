"""Load raw CSV, preprocess, and save the processed dataset."""

import pandas as pd
import logging
from pathlib import Path

from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw credit risk CSV."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps."""
    initial = len(df)
    logger.info("Starting cleaning pipeline")

    # Drop rows with any null values
    df = df.dropna()
    dropped = initial - len(df)
    logger.info(f"Dropped {dropped:,} rows with nulls → {len(df):,} remaining")

    # Filter unrealistic age values (data errors)
    df = df[df["person_age"] <= 100]
    logger.info(f"Filtered age > 100 → {len(df):,} remaining")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns."""
    logger.info("Encoding categoricals")

    # Binary encode cb_person_default_on_file (Y=1, N=0)
    df["cb_person_default_on_file"] = (
        df["cb_person_default_on_file"].str.strip().str.lower() == "y"
    ).astype(int)

    # Ordinal encode loan_grade (A=1 … G=7)
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["loan_grade"] = df["loan_grade"].str.strip().str.upper().map(grade_map)

    # One-hot encode person_home_ownership and loan_intent (guard against missing cols)
    one_hot_cols = [c for c in ["person_home_ownership", "loan_intent"] if c in df.columns]
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    logger.info(f"After encoding: {df.shape[1]} columns")
    return df


def main():
    """Run the full pipeline: load → clean → encode → save."""
    raw_path = RAW_DATA_PATH
    out_path = PROCESSED_DATA_PATH

    df = load_raw(raw_path)
    df = clean(df)
    df = encode_categoricals(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed data to {out_path} — {len(df):,} rows × {df.shape[1]} columns")

    # Log basic stats
    logger.info(
        f"Default rate in processed data: {df['loan_status'].mean():.3f} "
        f"({df['loan_status'].sum():,} defaults / {len(df):,} total)"
    )


if __name__ == "__main__":
    main()
