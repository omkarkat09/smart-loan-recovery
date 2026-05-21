import pandas as pd
import os
import datetime
import json
from evidently.report import Report
from evidently.metrics import DataDriftTable, ColumnDriftMetric, DatasetMissingValuesSummary
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def detect_drift(reference_path: str, current_path: str, pushgateway_url: str = "localhost:9091"):
    """
    Loads reference and current parquet files, runs an Evidently drift report,
    saves it to an HTML file, and pushes the overall drift metric to a Prometheus Pushgateway.
    """
    print(f"Loading reference data from {reference_path}")
    ref_df = pd.read_parquet(reference_path)
    
    print(f"Loading current data from {current_path}")
    curr_df = pd.read_parquet(current_path)

    # Initialize the Evidently report with requested metrics
    report = Report(metrics=[
        DataDriftTable(),
        ColumnDriftMetric(column_name="credit_score"),
        ColumnDriftMetric(column_name="dti_ratio"),
        ColumnDriftMetric(column_name="days_overdue"),
        ColumnDriftMetric(column_name="payment_history_6m"),
        DatasetMissingValuesSummary()
    ])

    print("Running drift analysis...")
    report.run(reference_data=ref_df, current_data=curr_df)

    # Save HTML report
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f'drift_{date_str}.html')
    report.save_html(report_path)
    print(f"Report saved to {report_path}")

    # Extract results as dict
    report_dict = report.as_dict()
    
    # Find DataDriftTable result to get overall dataset drift
    drift_table_result = None
    for m in report_dict['metrics']:
        if m['metric'] == 'DataDriftTable':
            drift_table_result = m['result']
            break
            
    if drift_table_result is None:
        raise ValueError("DataDriftTable metric result not found in Evidently report")

    overall_drift_detected = drift_table_result['dataset_drift']
    n_drifted_features = drift_table_result['number_of_drifted_columns']
    
    # Extract per-feature drift scores
    per_feature_scores = {}
    for col, col_data in drift_table_result['drift_by_columns'].items():
        per_feature_scores[col] = {
            'drift_score': col_data['drift_score'],
            'drift_detected': col_data['drift_detected']
        }

    # Push to Prometheus Pushgateway
    registry = CollectorRegistry()
    drift_gauge = Gauge('slr_drift_score', 'Overall drift score (1.0 for drift, 0.0 for no drift)', registry=registry)
    
    score_val = 1.0 if overall_drift_detected else 0.0
    drift_gauge.set(score_val)
    
    try:
        push_to_gateway(pushgateway_url, job='evidently_drift_detection', registry=registry)
        print(f"Successfully pushed slr_drift_score={score_val} to {pushgateway_url}")
    except Exception as e:
        print(f"Warning: Failed to push to Prometheus pushgateway at {pushgateway_url}. Is it running? Error: {e}")

    return {
        'overall_drift_detected': overall_drift_detected,
        'n_drifted_features': n_drifted_features,
        'per_feature_scores': per_feature_scores,
        'report_path': report_path
    }

if __name__ == "__main__":
    # Self-test block: Generate mock 'current' data with deliberate drift and run the pipeline
    curr_data_path = "data/current_batch.parquet"
    import numpy as np
    
    n_samples = 1000
    np.random.seed(99)
    
    # Introducing drift specifically in credit_score and days_overdue
    data = {
        'customer_id': [f"CUST-NEW-{i:05d}" for i in range(n_samples)],
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'credit_score': np.random.normal(550, 50, n_samples).astype(int), # drifted (lower)
        'dti_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'days_overdue': np.random.exponential(30, n_samples).astype(int), # drifted (higher)
        'payment_history_6m': np.random.randint(0, 7, n_samples)
    }
    
    os.makedirs(os.path.dirname(curr_data_path), exist_ok=True)
    pd.DataFrame(data).to_parquet(curr_data_path, index=False)
    
    try:
        result = detect_drift("data/reference.parquet", curr_data_path)
        print("\nDrift detection pipeline execution completed:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nError executing drift detection: {e}")
