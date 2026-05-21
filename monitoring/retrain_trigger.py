import requests
import subprocess
import argparse
import logging
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMETHEUS_URL = "http://localhost:9090"
SLACK_WEBHOOK_URL = "http://slack-webhook-placeholder"
MODEL_NAME = "smart-loan-recovery-ensemble"

def query_prometheus(query, step="1h", duration="6h"):
    """Queries Prometheus for a metric over a specified duration."""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={
            'query': f"{query}[{duration}:{step}]"
        })
        if response.status_code == 200:
            return response.json().get('data', {}).get('result', [])
    except Exception as e:
        logger.error(f"Failed to query Prometheus: {e}")
    return []

def check_retrain_needed():
    """Checks if AUC is degraded or drift is detected persistently."""
    auc_data = query_prometheus("slr_prediction_auc")
    drift_data = query_prometheus("slr_drift_score")

    auc_degraded = False
    drift_detected = False

    if auc_data and auc_data[0].get('values'):
        values = [float(v[1]) for v in auc_data[0]['values']]
        # Check if the last 3 hourly readings are below 0.75
        if len(values) >= 3 and all(v < 0.75 for v in values[-3:]):
            auc_degraded = True
            logger.warning("AUC degraded below 0.75 for 3 consecutive hours.")

    if drift_data and drift_data[0].get('values'):
        values = [float(v[1]) for v in drift_data[0]['values']]
        # Check if the last 3 hourly readings have a drift score > 0.3
        if len(values) >= 3 and all(v > 0.3 for v in values[-3:]):
            drift_detected = True
            logger.warning("Drift score above 0.3 for 3 consecutive hours.")

    return auc_degraded or drift_detected

def run_retrain():
    """Executes the bash pipeline and evaluates the resulting MLflow run."""
    logger.info("Triggering retraining pipeline via monitoring/retrain_pipeline.sh")
    
    try:
        subprocess.run(["bash", "monitoring/retrain_pipeline.sh"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining pipeline failed: {e}")
        return

    client = MlflowClient()
    
    # 1. Get the current registered model in Staging
    current_staging = None
    current_auc = 0.0
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if versions:
            current_staging = versions[0]
            run = client.get_run(current_staging.run_id)
            current_auc = run.data.metrics.get('test_auc', run.data.metrics.get('val_auc', 0.0))
    except Exception as e:
        logger.warning(f"Could not fetch current staging model (it might not exist yet): {e}")

    # 2. Get the newly trained model (most recent run across all active experiments)
    experiments = client.search_experiments()
    exp_ids = [exp.experiment_id for exp in experiments]
    
    latest_runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not latest_runs:
        logger.error("No MLflow runs found after retraining.")
        return

    new_run = latest_runs[0]
    new_auc = new_run.data.metrics.get('test_auc', new_run.data.metrics.get('val_auc', 0.0))

    logger.info(f"Validation completed. Current AUC: {current_auc:.4f}, New AUC: {new_auc:.4f}")

    # 3. Compare and conditionally promote
    if new_auc > current_auc:
        logger.info("New model improved AUC. Registering and transitioning to Staging.")
        try:
            # Register the new model version
            new_version = client.create_model_version(
                name=MODEL_NAME, 
                source=new_run.info.artifact_uri, 
                run_id=new_run.info.run_id
            )
            # Promote to Staging
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version.version,
                stage="Staging",
                archive_existing_versions=True
            )
            message = f"✅ *Automated Retraining Success!*\nNew model AUC (`{new_auc:.4f}`) improved over previous AUC (`{current_auc:.4f}`). The model has been promoted to Staging."
        except Exception as e:
            logger.error(f"Failed to register/transition model: {e}")
            message = f"❌ *Automated Retraining Error*\nFailed to register improved model. See logs for details."
    else:
        logger.warning("New model did NOT improve AUC. Keeping current model.")
        message = f"⚠️ *Automated Retraining Warning*\nNew model AUC (`{new_auc:.4f}`) is worse or equal to the previous AUC (`{current_auc:.4f}`). The new model was discarded."

    # Send Slack Notification
    try:
        requests.post(SLACK_WEBHOOK_URL, json={'text': message})
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Loan Recovery Automated Retraining Trigger")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of current Prometheus metrics")
    args = parser.parse_args()

    if args.force:
        logger.info("Forced retrain triggered via CLI.")
        run_retrain()
    elif check_retrain_needed():
        logger.info("Retrain triggered via automated Prometheus metric thresholds.")
        run_retrain()
    else:
        logger.info("Metrics are healthy. No retraining necessary at this time.")
