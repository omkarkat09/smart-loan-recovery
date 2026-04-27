"""Feature engineering — Phase 2 Step 2.

Build four groups of features:
  1. Temporal         — days_since_payment, rolling_30d_sum
  2. Behavioral       — avg_payment_delay, missed_pay_ratio
  3. Credit Risk      — dti_ratio, ltv_ratio, credit_utilization
  4. Aggregated       — customer_totals (tier aggregates), max_delinquency

Since the dataset has loan-level data but no individual payment ledger,
behavioural / temporal features are derived from loan attributes and
plausible proxy signals rather than true time-series payment records.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from config.settings import PROCESSED_DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    days_since_payment
        Proxy: months since credit history started (inverse proxy for recency).
        Larger cb_person_cred_hist_length → older file → less recent activity.
        We derive days_approx = (credit_hist_months * 30) as a relative baseline.

    rolling_30d_sum
        Proxy: modelled payment intensity as a function of loan amount,
        interest rate, and install frequency.
        Higher rolling_30d_sum → borrower is managing larger/ more loans.
    """
    logger.info("Adding temporal features")

    # cb_person_cred_hist_length is in years → convert to days approx
    df["credit_hist_days"] = (df["cb_person_cred_hist_length"] * 365).astype("int32")

    # Proxy for "days since last payment activity" — inverse of credit history age
    # (younger credit file = more recent account opening = more recent activity)
    # We clip so it never goes negative and cap at 15 years
    df["days_since_payment_proxy"] = (
        (15 * 365) - (df["cb_person_cred_hist_length"] * 365).clip(upper=15 * 365)
    ).astype("int32")

    # rolling_30d_sum proxy: modelled monthly payment burden
    # installment ≈ loan_amnt * (rate/12) approximation; here we use loan_amnt + int_burden
    monthly_installment_approx = (
        df["loan_amnt"] * (df["loan_int_rate"] / 100) / 12
    ).clip(lower=1)

    # Rolling 30d sum: weighted by number of active credit lines (proxied by loan_grade severity)
    # Grade A (1) borrowers → lower risk → assumed active on fewer large loans
    # Grade G (7) borrowers → higher risk → assumed multiple small loans
    grade_activity_weight = df["loan_grade"].clip(lower=1)
    df["rolling_30d_sum"] = (monthly_installment_approx * grade_activity_weight).astype("float32")

    logger.info(
        f"  temporal: credit_hist_days range [{df['credit_hist_days'].min():,}, "
        f"{df['credit_hist_days'].max():,}] | "
        f"rolling_30d_sum range [{df['rolling_30d_sum'].min():.0f}, "
        f"{df['rolling_30d_sum'].max():.0f}]"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. BEHAVIORAL
# ─────────────────────────────────────────────────────────────────────────────

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    avg_payment_delay
        Proxy: modelled average delay tendency based on interest rate,
        loan grade, and employment length.
        High rate + long employment gap → higher assumed delay.

    missed_pay_ratio
        Binary indicator: borrower shows multiple risk signals simultaneously.
        cb_person_default_on_file=1 AND dti high AND grade D+ → high missed ratio.
        Also derived: loan_percent_income > 30% with interest rate > 15% → flagged.
    """
    logger.info("Adding behavioural features")

    # avg_payment_delay: multi-signal score → mapped to a [0, 1] normalised delay score
    delay_raw = (
        df["loan_int_rate"] / 25.0                           # normalised rate contribution
        + (df["loan_grade"] - 1) / 6.0                        # normalised grade (A=0, G=1)
        - (df["person_emp_length"].clip(upper=20) / 20.0)   # employment stability reduces delay
    )
    df["avg_payment_delay"] = delay_raw.clip(0, 1).astype("float32")

    # missed_pay_ratio: multi-risk-signal flag
    high_dti     = (df["loan_percent_income"] > 20).astype("float32")
    high_rate    = (df["loan_int_rate"] > 14).astype("float32")
    prior_default = df["cb_person_default_on_file"].astype("float32")
    high_grade   = (df["loan_grade"] >= 5).astype("float32")   # E, F, G

    df["missed_pay_ratio"] = (
        (high_dti + high_rate + prior_default + high_grade) / 4.0
    ).astype("float32")

    logger.info(
        f"  behavioural: avg_payment_delay [{df['avg_payment_delay'].min():.2f}, "
        f"{df['avg_payment_delay'].max():.2f}] | "
        f"missed_pay_ratio [{df['missed_pay_ratio'].min():.2f}, "
        f"{df['missed_pay_ratio'].max():.2f}]"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CREDIT RISK RATIOS
# ─────────────────────────────────────────────────────────────────────────────

def add_credit_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    dti_ratio        — Debt-to-income: loan_percent_income already captures this,
                        but we also compute a combined ratio using loan_int_rate.

    ltv_ratio        — Loan-to-Value: loan_amnt relative to annual_income.
                        High LTV (>0.5) means loan is >50% of annual income → risky.

    credit_utilization — Proxy: ratio of loan burden to credit history quality.
                        More credit history + high loan → lower utilisation signal quality.
                        Derived: loan_amnt / (income * credit_hist_years_normalised)
    """
    logger.info("Adding credit risk features")

    # dti_ratio: enhanced ratio using both loan_percent_income and interest burden
    interest_burden = df["loan_int_rate"] / 100 * df["loan_amnt"] / df["person_income"].clip(lower=1)
    df["dti_ratio"] = (df["loan_percent_income"] / 100 + interest_burden).clip(0, 1).astype("float32")

    # ltv_ratio: loan amount as fraction of annual income
    df["ltv_ratio"] = (df["loan_amnt"] / df["person_income"].clip(lower=1)).clip(0, 3).astype("float32")

    # credit_utilization: loan relative to income-adjusted credit quality
    # Normalised credit history quality: older file + higher income = better utilisation signal
    credit_quality = (df["cb_person_cred_hist_length"] * df["person_income"] / 1e6).clip(1, None)
    df["credit_utilization"] = (df["loan_amnt"] / credit_quality).astype("float32")

    logger.info(
        f"  credit risk: dti_ratio [{df['dti_ratio'].min():.2f}, {df['dti_ratio'].max():.2f}] | "
        f"ltv_ratio [{df['ltv_ratio'].min():.2f}, {df['ltv_ratio'].max():.2f}] | "
        f"credit_utilization [{df['credit_utilization'].min():.1f}, {df['credit_utilization'].max():.1f}]"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGGREGATED
# ─────────────────────────────────────────────────────────────────────────────

def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    customer_totals
        Group-level aggregates by loan_grade tier and employment stability group.
        - avg_loan_by_grade     : mean loan_amnt per loan_grade
        - max_loan_by_grade     : max loan_amnt per loan_grade
        - income_score_by_grade : mean income per loan_grade
        - emp_stability_score   : normalised employment length per income bracket

    max_delinquency
        Proxy: highest risk tier the borrower falls into based on combination of:
        - grade >= 6 (F/G), interest > 18%, income < $30k, prior default, high DTI
        Each condition adds 1 to a delinquency tier score [0-5].
    """
    logger.info("Adding aggregated features")

    # Customer totals — group statistics by loan_grade
    grade_loan_avg = df.groupby("loan_grade")["loan_amnt"].transform("mean")
    df["avg_loan_by_grade"] = grade_loan_avg.astype("float32")

    grade_loan_max = df.groupby("loan_grade")["loan_amnt"].transform("max")
    df["max_loan_by_grade"] = grade_loan_max.astype("float32")

    grade_income_avg = df.groupby("loan_grade")["person_income"].transform("mean")
    df["income_score_by_grade"] = grade_income_avg.astype("float32")

    # Employment stability: normalised by income bracket
    income_bracket = pd.qcut(df["person_income"], q=4, labels=["low", "mid", "high", "vhigh"])
    df["emp_stability_score"] = (
        df["person_emp_length"] / df["person_income"] * 1e5
    ).astype("float32")

    # max_delinquency tier score
    delinquency_conditions = pd.DataFrame({
        "grade_D_plus":     (df["loan_grade"] >= 4).astype("int8"),   # D or worse
        "grade_FG":         (df["loan_grade"] >= 6).astype("int8"),   # F or G
        "high_rate":        (df["loan_int_rate"] > 16).astype("int8"),
        "low_income":       (df["person_income"] < 30_000).astype("int8"),
        "prior_default":    df["cb_person_default_on_file"].astype("int8"),
        "high_dti":         (df["loan_percent_income"] > 25).astype("int8"),
    })

    df["max_delinquency"] = delinquency_conditions.sum(axis=1).astype("int8")
    df["max_delinquency"] = df["max_delinquency"].clip(0, 5)

    logger.info(
        f"  aggregated: avg_loan_by_grade [{df['avg_loan_by_grade'].min():.0f}, "
        f"{df['avg_loan_by_grade'].max():.0f}] | "
        f"max_delinquency distribution:\n"
        f"  {df['max_delinquency'].value_counts().sort_index().to_dict()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all four feature groups in sequence."""
    logger.info(f"Building features on {len(df):,} rows")
    df = add_temporal_features(df)
    df = add_behavioral_features(df)
    df = add_credit_risk_features(df)
    df = add_aggregated_features(df)

    new_cols = [c for c in df.columns if c not in [
        "person_age", "person_income", "person_emp_length", "loan_grade", "loan_amnt",
        "loan_int_rate", "loan_status", "loan_percent_income", "cb_person_default_on_file",
        "cb_person_cred_hist_length", "person_home_ownership_Other", "person_home_ownership_Own",
        "person_home_ownership_Rent", "loan_intent_Education", "loan_intent_Homeimprovement",
        "loan_intent_Medical", "loan_intent_Personal", "loan_intent_Venture"
    ]]
    logger.info(f"Feature engineering complete — {len(new_cols)} new features added: {new_cols}")
    return df


def main():
    """Load processed data, build features, save enriched dataset."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols from processed data")

    df = build_features(df)

    out_path = PROCESSED_DATA_PATH
    df.to_csv(out_path, index=False)
    logger.info(
        f"Saved enriched dataset → {out_path} | "
        f"{len(df):,} rows × {df.shape[1]} cols | "
        f"Default rate: {df['loan_status'].mean():.1%}"
    )
    logger.info(f"Final columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()