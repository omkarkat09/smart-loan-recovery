"""Unit tests for the data processing pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.make_dataset import clean, encode_categoricals


@pytest.fixture
def raw_df():
    """Minimal DataFrame mimicking raw credit_risk_dataset.csv structure."""
    return pd.DataFrame({
        "person_age":                  [25, 30, 67, 22, 999],          # 999 = bad age
        "person_income":               [40000, 55000, 80000, 30000, 45000],
        "person_home_ownership":       ["RENT", "OWN", "MORTGAGE", "RENT", "RENT"],
        "person_emp_length":           [2.0, 5.0, np.nan, 1.0, 3.0],    # one null
        "loan_intent":                 ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION"],
        "loan_grade":                  ["A", "B", "C", "D", "E"],
        "loan_amnt":                   [5000, 8000, 12000, 35000, 25000],
        "loan_int_rate":               [6.0, 8.5, 11.0, 14.5, 18.0],
        "loan_status":                 [0, 0, 1, 1, 1],                  # 2 non-default, 3 default
        "loan_percent_income":         [12.5, 14.5, 15.0, 40.0, 55.6],
        "cb_person_default_on_file":    ["N", "N", "Y", "N", "Y"],
        "cb_person_cred_hist_length":  [3, 5, 8, 2, 4],
    })


class TestClean:
    def test_dropna_removes_nulls(self, raw_df):
        df = clean(raw_df)
        assert df.isnull().sum().sum() == 0, "No null values should remain"

    def test_filter_age_outliers(self, raw_df):
        df = clean(raw_df)
        assert df["person_age"].max() <= 100, "Ages above 100 should be filtered"

    def test_preserves_target(self, raw_df):
        df = clean(raw_df)
        assert "loan_status" in df.columns
        assert df["loan_status"].isin([0, 1]).all(), "Target must be binary 0/1"

    def test_default_rate_preserved(self, raw_df):
        """After cleaning, the default rate should still be computable."""
        df = clean(raw_df)
        rate = df["loan_status"].mean()
        assert 0.0 <= rate <= 1.0, "Default rate must be between 0 and 1"


class TestEncodeCategoricals:
    def test_binary_encoding(self):
        df = pd.DataFrame({
            "cb_person_default_on_file": ["Y", "N", "Y", "N"],
            "loan_grade": ["A", "B", "C", "D"],
            "loan_status": [1, 0, 1, 0],
        })
        result = encode_categoricals(df)
        assert result["cb_person_default_on_file"].dtype in [int, bool], \
            "cb_person_default_on_file should be numeric after encoding"

    def test_grade_ordinal(self):
        df = pd.DataFrame({
            "cb_person_default_on_file": ["N", "N"],
            "loan_grade": ["A", "G"],
            "loan_status": [0, 1],
        })
        result = encode_categoricals(df)
        assert result.loc[result["loan_grade"] == result["loan_grade"].min(), "loan_grade"].iloc[0] == 1
        assert result["loan_grade"].max() == 7, "loan_grade should range 1–7"

    def test_one_hot_columns_created(self):
        df = pd.DataFrame({
            "person_home_ownership": ["RENT", "OWN", "MORTGAGE"],
            "loan_intent":            ["EDUCATION", "MEDICAL", "PERSONAL"],
            "cb_person_default_on_file": ["N", "N", "N"],
            "loan_grade": ["A", "B", "C"],
            "loan_status": [0, 0, 1],
        })
        result = encode_categoricals(df)
        for col in ["person_home_ownership", "loan_intent"]:
            assert any(col in c for c in result.columns), f"Expected one-hot columns for {col}"

    def test_encode_categoricals_runs_without_error(self):
        """Smoke test: encode_categoricals should run without KeyError on any input df."""
        df = pd.DataFrame({
            "cb_person_default_on_file": ["Y", "N"],
            "loan_grade": ["A", "G"],
            "loan_status": [1, 0],
        })
        result = encode_categoricals(df)
        assert result is not None
        assert len(result) == 2
