import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import impute_nulls, cap_outliers, fix_dtypes, encode_categoricals
from src.features.build_features import build_features
from src.data.pipeline import add_customer_id

def create_mock_raw_data(n_rows=10, time_col=False):
    """Creates a synthetic raw dataset that mimics the loan recovery CSV."""
    np.random.seed(42)
    data = {
        "person_age": np.random.randint(20, 60, n_rows),
        "person_income": np.random.randint(20000, 100000, n_rows),
        "person_emp_length": np.random.randint(0, 15, n_rows),
        "loan_grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_rows),
        "loan_amnt": np.random.randint(1000, 30000, n_rows),
        "loan_int_rate": np.random.uniform(5.0, 20.0, n_rows),
        "loan_status": np.random.choice([0, 1], n_rows),
        "loan_percent_income": np.random.uniform(1.0, 30.0, n_rows),
        "cb_person_default_on_file": np.random.choice(["Y", "N"], n_rows),
        "cb_person_cred_hist_length": np.random.randint(1, 30, n_rows),
        "person_home_ownership": np.random.choice(["RENTED", "OWN", "OTHER"], n_rows),
        "loan_intent": np.random.choice(["PERSONAL", "EDUCATION", "HOMEIMPROVEMENT", "VENTURE", "MEDICAL"], n_rows),
    }
    df = pd.DataFrame(data)
    if time_col:
        df["timestamp"] = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                                         "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"][:n_rows])
    return df

def run_full_pipeline_on_df(df):
    """Helper to run the data processing stages on a dataframe."""
    df = df.copy()
    df = impute_nulls(df)
    df = cap_outliers(df)
    df = fix_dtypes(df)
    df = encode_categoricals(df)
    df = build_features(df)
    df = add_customer_id(df)
    return df

def test_feature_columns_exist():
    """Test 1: Verify all expected feature columns exist in the output."""
    df_raw = create_mock_raw_data()
    df_processed = run_full_pipeline_on_df(df_raw)

    expected_features = [
        "credit_hist_days", "days_since_payment_proxy", "rolling_30d_sum",
        "avg_payment_delay", "missed_pay_ratio",
        "dti_ratio", "ltv_ratio", "credit_utilization",
        "avg_loan_by_grade", "max_loan_by_grade", "income_score_by_grade",
        "emp_stability_score", "max_delinquency", "customer_id"
    ]

    for col in expected_features:
        assert col in df_processed.columns, f"Expected column {col} missing from processed data"

def test_no_nans_after_pipeline():
    """Test 2: Verify there are no NaN values after the pipeline runs."""
    # Create data with some NaNs to test imputation
    df_raw = create_mock_raw_data(n_rows=20)
    df_raw.loc[0, "person_emp_length"] = np.nan
    df_raw.loc[1, "loan_int_rate"] = np.nan
    df_raw.loc[2, "person_home_ownership"] = np.nan

    df_processed = run_full_pipeline_on_df(df_raw)

    assert df_processed.isnull().sum().sum() == 0, "Pipeline left NaN values in the output"

def test_no_future_leakage():
    """
    Test 3: Verify no feature was accidentally computed using future data.
    Feeds data in chronological order and asserts that the feature value at
    time T is the same whether we have data up to T or data up to T+N.
    """
    # We need at least 2 rows with same grade to test aggregates
    # Force some same grades
    df_raw = create_mock_raw_data(n_rows=10, time_col=True)
    df_raw["loan_grade"] = "A"
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    # T is the index we want to check for leakage
    t_index = 5

    # Scenario A: Pipeline runs on data up to T
    df_until_t = df_raw.iloc[:t_index + 1].copy()
    df_processed_a = run_full_pipeline_on_df(df_until_t)
    val_a = df_processed_a.iloc[t_index].copy()

    # Scenario B: Pipeline runs on data up to T + N (the whole set)
    df_processed_b = run_full_pipeline_on_df(df_raw)
    val_b = df_processed_b.iloc[t_index].copy()

    # Compare all columns that are likely to be aggregated/rolling
    leakage_candidates = [
        "avg_loan_by_grade", "max_loan_by_grade", "income_score_by_grade",
        "rolling_30d_sum", "max_delinquency"
    ]

    for col in leakage_candidates:
        if col in val_a.index:
            # Use np.isclose for floats
            assert np.isclose(val_a[col], val_b[col]), f"Data leakage detected in feature: {col}. Value at index {t_index} changed when future data was added."

if __name__ == "__main__":
    pytest.main([__file__])
