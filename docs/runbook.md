# Smart Loan Recovery - Operational Runbook

This runbook outlines the six primary operational procedures for managing the Smart Loan Recovery ML system in production.

## 1. Restart the API Pods
If the FastAPI service enters a degraded state (e.g., deadlock, memory leak) and needs a forced restart without changing the deployment configuration, perform a rolling restart:
```bash
kubectl rollout restart deployment/loan-recovery-api
```
To monitor the status of the rollout:
```bash
kubectl rollout status deployment/loan-recovery-api
```

## 2. Roll Back to a Previous Model
If a newly promoted model severely degrades production metrics, roll back to the previously stable version using the MLflow CLI.

First, identify the correct version number from the MLflow UI, then transition it back to Staging (which the API pods will automatically pull if they are restarted, or if polling is enabled):
```bash
# Example: Rolling back to Version 3 of the model
export MLFLOW_TRACKING_URI="http://localhost:5000"
mlflow models transition-stage --name "smart-loan-recovery-ensemble" --version 3 --stage "Staging" --archive-existing-versions
```
*(After transitioning, restart the API pods using procedure #1 to force them to load the reverted model into memory).*

## 3. Run a Manual Drift Check
If you suspect recent data anomalies and don't want to wait for the scheduled cron job, you can trigger the Evidently AI drift check manually:
```bash
# Ensure your environment variables and virtual env are loaded
python monitoring/drift_detection.py
```
This will:
- Compare `data/current_batch.parquet` against `data/reference.parquet`.
- Generate an HTML report in `monitoring/reports/`.
- Push the resulting `slr_drift_score` to Prometheus.

## 4. Trigger a Manual Retrain
If a drift alert fires or AUC is persistently degraded, but the automated retrain trigger failed or wasn't scheduled yet, force a retraining cycle:
```bash
python monitoring/retrain_trigger.py --force
```
Using `--force` bypasses the Prometheus threshold checks, directly launching the data processing, feature engineering, and model training pipelines. It will safely compare the resulting model against the current Staging model before promoting.

## 5. Silence a Prometheus Alert
If an alert is firing expectedly (e.g., during planned maintenance or a known upstream data outage), you can silence it using the Alertmanager CLI tool (`amtool`):
```bash
# Silence the DataDriftDetected alert for 4 hours
amtool silence add alertname="DataDriftDetected" \
  --duration="4h" \
  --author="ops-engineer" \
  --comment="Known upstream data schema change, retraining pipeline is currently running." \
  --alertmanager.url="http://localhost:9093"
```

## 6. Access the Grafana Dashboard
To view the Smart Loan Recovery dashboard directly from a Kubernetes cluster where LoadBalancers or Ingresses are not yet publicly exposed, use port-forwarding:
```bash
# Forward local port 3000 to the Grafana pod/service
kubectl port-forward svc/grafana 3000:3000
```
Open your browser and navigate to:
**[http://localhost:3000](http://localhost:3000)**
*(Default credentials are typically `admin` / `admin` unless configured otherwise via secrets).*
