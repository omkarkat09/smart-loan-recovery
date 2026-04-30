"""Unit tests for the feature engineering pipeline (src/features/build_features.py)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import (
    add_temporal_features,
    add_behavioral_features,
    add_credit_risk_features,
    add_aggregated_features,
    build_features,
)


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_df():
    """Minimal DataFrame with all required base columns."""
    return pd.DataFrame({
        "person_age":                  [25, 30, 45, 22, 38],
        "person_income":               [40000, 80000, 55000, 25000, 95000],
        "person_emp_length":           [2.0, 8.0, 5.0, 1.0, 15.0],
        "loan_grade":                  [2, 1, 4, 6, 3],        # B, A, D, F, C
        "loan_amnt":                   [5000, 8000, 15000, 30000, 12000],
        "loan_int_rate":               [11.0, 6.5, 14.5, 20.0, 9.5],
        "loan_status":                  [0, 0, 1, 1, 0],
        "loan_percent_income":          [12.5, 10.0, 27.3, 50.0, 12.6],
        "cb_person_default_on_file":    [0, 0, 1, 1, 0],
        "cb_person_cred_hist_length":   [3, 10, 5, 2, 8],
        "person_home_ownership_Other": [0, 0, 0, 0, 0],
        "person_home_ownership_Own":   [0, 1, 0, 0, 0],
        "person_home_ownership_Rent":  [1, 0, 1, 1, 0],
        "loan_intent_Education":        [1, 0, 0, 0, 1],
        "loan_intent_Homeimprovement":  [0, 0, 1, 0, 0],
        "loan_intent_Medical":          [0, 1, 0, 0, 0],
        "loan_intent_Personal":         [0, 0, 0, 0, 0],
        "loan_intent_Venture":          [0, 0, 0, 1, 0],
    })


# ── Temporal ───────────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_credit_hist_days_in_range(self, base_df):
        df = add_temporal_features(base_df.copy())
        assert df["credit_hist_days"].between(0, 15 * 365).all()

    def test_days_since_payment_proxy_non_negative(self, base_df):
        df = add_temporal_features(base_df.copy())
        assert (df["days_since_payment_proxy"] >= 0).all()

    def test_rolling_30d_sum_positive(self, base_df):
        df = add_temporal_features(base_df.copy())
        assert (df["rolling_30d_sum"] > 0).all()

    def test_no_nulls_in_temporal_features(self, base_df):
        df = add_temporal_features(base_df.copy())
        for col in ["credit_hist_days", "days_since_payment_proxy", "rolling_30d_sum"]:
            assert df[col].notna().all(), f"{col} should not have nulls"


# ── Behavioural ────────────────────────────────────────────────────────────────

class TestBehaviouralFeatures:
    def test_avg_payment_delay_range(self, base_df):
        df = add_behavioral_features(base_df.copy())
        assert df["avg_payment_delay"].between(0, 1).all()

    def test_missed_pay_ratio_range(self, base_df):
        df = add_behavioral_features(base_df.copy())
        assert df["missed_pay_ratio"].between(0, 1).all()

    def test_no_nulls_in_behavioural_features(self, base_df):
        df = add_behavioral_features(base_df.copy())
        for col in ["avg_payment_delay", "missed_pay_ratio"]:
            assert df[col].notna().all()

    def test_prior_default_increases_missed_ratio(self, base_df):
        df = add_behavioral_features(base_df.copy())
        # Row 2 (index=2): prior_default=1, grade=4, high rate → should score higher
        # Row 0 (index=0): no risk factors → should score lower
        assert df.loc[2, "missed_pay_ratio"] > df.loc[0, "missed_pay_ratio"]


# ── Credit Risk ─────────────────────────────────────────────────────────────────

class TestCreditRiskFeatures:
    def test_dti_ratio_range(self, base_df):
        df = add_credit_risk_features(base_df.copy())
        assert df["dti_ratio"].between(0, 1).all(), "dti_ratio should be in [0, 1]"

    def test_ltv_ratio_positive(self, base_df):
        df = add_credit_risk_features(base_df.copy())
        assert (df["ltv_ratio"] >= 0).all()

    def test_credit_utilization_positive(self, base_df):
        df = add_credit_risk_features(base_df.copy())
        assert (df["credit_utilization"] > 0).all()

    def test_no_nulls_in_credit_risk_features(self, base_df):
        df = add_credit_risk_features(base_df.copy())
        for col in ["dti_ratio", "ltv_ratio", "credit_utilization"]:
            assert df[col].notna().all(), f"{col} should not have nulls"


# ── Aggregated ───────────────────────────────────────────────────────────────────

class TestAggregatedFeatures:
    def test_avg_loan_by_grade_grouped(self, base_df):
        df = add_aggregated_features(base_df.copy())
        # Each row's avg_loan_by_grade should equal the mean for its grade
        for grade in df["loan_grade"].unique():
            mask = df["loan_grade"] == grade
            expected = df.loc[mask, "loan_amnt"].mean()
            assert df.loc[mask, "avg_loan_by_grade"].round(2).nunique() == 1
            assert abs(df.loc[mask, "avg_loan_by_grade"].iloc[0] - expected) < 0.01

    def test_max_delinquency_in_valid_range(self, base_df):
        df = add_aggregated_features(base_df.copy())
        assert df["max_delinquency"].between(0, 5).all()

    def test_max_delinquency_increases_with_risk(self, base_df):
        df = add_aggregated_features(base_df.copy())
        # Row with grade=6 (F), rate=20, prior_default=1, high DTI should score >= row with A-grade, no risk
        # Row 0: grade=2, rate=11, no prior default, DTI=12.5 → low risk
        # Row 3: grade=6, rate=20, prior_default=1, DTI=50 → highest risk
        assert df.loc[3, "max_delinquency"] > df.loc[0, "max_delinquency"]

    def test_emp_stability_score_positive(self, base_df):
        df = add_aggregated_features(base_df.copy())
        assert (df["emp_stability_score"] >= 0).all()


# ── build_features (all groups) ─────────────────────────────────────────────────

class TestBuildFeatures:
    def test_all_new_features_present(self, base_df):
        df = build_features(base_df.copy())
        new_features = [
            "credit_hist_days", "days_since_payment_proxy", "rolling_30d_sum",
            "avg_payment_delay", "missed_pay_ratio",
            "dti_ratio", "ltv_ratio", "credit_utilization",
            "avg_loan_by_grade", "max_loan_by_grade", "income_score_by_grade",
            "emp_stability_score", "max_delinquency",
        ]
        for feat in new_features:
            assert feat in df.columns, f"{feat} should be in DataFrame"

    def test_target_unchanged(self, base_df):
        df = build_features(base_df.copy())
        assert list(df["loan_status"]) == list(base_df["loan_status"])

    def test_original_features_unchanged(self, base_df):
        df = build_features(base_df.copy())
        for col in ["person_income", "loan_grade", "loan_amnt", "loan_int_rate"]:
            assert df[col].equals(base_df[col]), f"{col} should not change"

    def test_no_nulls_in_new_features(self, base_df):
        df = build_features(base_df.copy())
        new_features = [
            "credit_hist_days", "days_since_payment_proxy", "rolling_30d_sum",
            "avg_payment_delay", "missed_pay_ratio",
            "dti_ratio", "ltv_ratio", "credit_utilization",
            "avg_loan_by_grade", "max_loan_by_grade", "income_score_by_grade",
            "emp_stability_score", "max_delinquency",
        ]
        for feat in new_features:
            assert df[feat].notna().all(), f"{feat} should have no nulls"