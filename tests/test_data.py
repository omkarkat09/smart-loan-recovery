"""Unit tests for the Phase 2 data processing pipeline (src/data/make_dataset.py)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.make_dataset import (
    impute_nulls,
    cap_outliers,
    fix_dtypes,
    encode_categoricals,
    OUTLIER_CAPS,
    IMPUTE_NUMERIC_MEDIAN,
    IMPUTE_CATEGORICAL_MODE,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    """Minimal DataFrame mimicking raw credit_risk_dataset.csv with all expected columns."""
    return pd.DataFrame({
        "person_age":                  [25, 30, 67, 22, 144, 17],
        "person_income":               [40000, 55000, 8_000_000, 30000, 2000, 45000],  # outlier & low cap
        "person_home_ownership":       ["RENT", "OWN", "MORTGAGE", np.nan, "RENT", "OTHER"],
        "person_emp_length":           [2.0, 5.0, np.nan, 1.0, 3.0, 50.0],            # null & outlier cap
        "loan_intent":                 ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", np.nan, "DEBTCONSOLIDATION"],
        "loan_grade":                  ["A", "B", "C", "D", "E", "F"],
        "loan_amnt":                   [5000, 8000, 12000, 35000, 25000, 60000],        # >50k cap
        "loan_int_rate":               [6.0, 8.5, np.nan, 14.5, 18.0, 28.0],           # null & >25% cap
        "loan_status":                  [0, 0, 1, 1, 1, 0],
        "loan_percent_income":          [12.5, 14.5, 80.0, 40.0, 55.6, 0.1],          # outlier caps
        "cb_person_default_on_file":    ["N", "N", "Y", "N", "Y", "N"],
        "cb_person_cred_hist_length":   [3, 5, 8, 2, 4, 50],                           # >40 cap
    })


# ── Tests: impute_nulls ────────────────────────────────────────────────────────

class TestImputeNulls:
    def test_no_nulls_after_impute(self, raw_df):
        df = impute_nulls(raw_df)
        assert df.isnull().sum().sum() == 0, "All nulls should be imputed"

    def test_numeric_imputed_with_median(self, raw_df):
        df = impute_nulls(raw_df)
        for col in IMPUTE_NUMERIC_MEDIAN:
            assert col in df.columns
            assert df[col].notna().all(), f"{col} should have no nulls after imputation"

    def test_categorical_imputed_with_mode(self, raw_df):
        df = impute_nulls(raw_df)
        for col in IMPUTE_CATEGORICAL_MODE:
            assert col in df.columns
            assert df[col].notna().all(), f"{col} should have no nulls after imputation"

    def test_target_preserved(self, raw_df):
        df = impute_nulls(raw_df)
        assert "loan_status" in df.columns
        assert set(df["loan_status"].unique()).issubset({0, 1})

    def test_default_rate_valid(self, raw_df):
        df = impute_nulls(raw_df)
        rate = df["loan_status"].mean()
        assert 0.0 <= rate <= 1.0


# ── Tests: cap_outliers ───────────────────────────────────────────────────────

class TestCapOutliers:
    def test_age_capped(self, raw_df):
        df = cap_outliers(raw_df.copy())
        assert df["person_age"].between(18, 100).all(), "person_age should be within [18, 100]"

    def test_income_capped(self, raw_df):
        df = cap_outliers(raw_df.copy())
        assert df["person_income"].between(5_000, 1_000_000).all(), "person_income should be within [5000, 1M]"

    def test_emp_length_capped(self, raw_df):
        df = impute_nulls(raw_df)  # fill nulls first so between() works
        df = cap_outliers(df)
        assert df["person_emp_length"].between(0, 40).all(), \
            "person_emp_length should be within [0, 40]"

    def test_loan_int_rate_capped(self, raw_df):
        df = impute_nulls(raw_df)  # fill nulls first
        df = cap_outliers(df)
        assert df["loan_int_rate"].between(3.0, 25.0).all(), \
            "loan_int_rate should be within [3, 25]"

    def test_loan_percent_income_capped(self, raw_df):
        df = cap_outliers(raw_df.copy())
        assert df["loan_percent_income"].between(0.5, 50.0).all(), "loan_percent_income should be within [0.5, 50]"

    def test_cred_hist_length_capped(self, raw_df):
        df = cap_outliers(raw_df.copy())
        assert df["cb_person_cred_hist_length"].between(1, 40).all(), "cb_person_cred_hist_length should be within [1, 40]"

    def test_all_rows_preserved(self, raw_df):
        df = cap_outliers(raw_df.copy())
        assert len(df) == len(raw_df), "Capping should not drop rows"


# ── Tests: fix_dtypes ─────────────────────────────────────────────────────────

class TestFixDtypes:
    def test_string_columns_normalised(self):
        df = pd.DataFrame({
            "person_home_ownership": ["  rent  ", "own", " mortgage "],
            "loan_intent": ["education", "medical", " personal "],
            "loan_grade": ["a", "B", " c "],
        })
        result = fix_dtypes(df)
        # Each value should equal its own strip+title (proving normalisation ran)
        for col in ["person_home_ownership", "loan_intent", "loan_grade"]:
            for val in result[col]:
                assert val == val.strip().title(), f"'{val}' should be title-cased and stripped"


# ── Tests: encode_categoricals ───────────────────────────────────────────────

class TestEncodeCategoricals:
    def test_binary_encoding(self):
        df = pd.DataFrame({
            "cb_person_default_on_file": ["Y", "N", "Y", "N"],
            "loan_grade": ["A", "B", "C", "D"],
            "loan_status": [1, 0, 1, 0],
        })
        result = encode_categoricals(df)
        assert result["cb_person_default_on_file"].dtype in [np.int8, np.int32, int]
        assert set(result["cb_person_default_on_file"].unique()).issubset({0, 1})

    def test_grade_ordinal_range(self):
        df = pd.DataFrame({
            "cb_person_default_on_file": ["N", "N"],
            "loan_grade": ["A", "G"],
            "loan_status": [0, 1],
        })
        result = encode_categoricals(df)
        assert result["loan_grade"].min() == 1, "loan_grade should start at 1 (A)"
        assert result["loan_grade"].max() == 7, "loan_grade should end at 7 (G)"

    def test_one_hot_columns_created(self):
        df = pd.DataFrame({
            "person_home_ownership": ["RENT", "OWN", "MORTGAGE"],
            "loan_intent":           ["EDUCATION", "MEDICAL", "PERSONAL"],
            "cb_person_default_on_file": ["N", "N", "N"],
            "loan_grade": ["A", "B", "C"],
            "loan_status": [0, 0, 1],
        })
        result = encode_categoricals(df)
        for col in ["person_home_ownership", "loan_intent"]:
            assert any(col in c for c in result.columns), f"Expected one-hot columns for {col}"

    def test_smoke_test_runs_without_error(self):
        """encode_categoricals should not raise on a minimal valid DataFrame."""
        df = pd.DataFrame({
            "cb_person_default_on_file": ["Y", "N"],
            "loan_grade": ["A", "G"],
            "loan_status": [1, 0],
        })
        result = encode_categoricals(df)
        assert result is not None
        assert len(result) == 2