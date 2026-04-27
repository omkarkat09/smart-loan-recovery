"""Train a logistic regression baseline and log metrics to MLflow."""

import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from config.settings import PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MLflow tracking
MLFLOW_TRACKING_URI = "file:///C:/Users/dell/GitHub/smart-loan-recovery/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("smart-loan-recovery-baseline")


def load_processed(path: str) -> pd.DataFrame:
    logger.info(f"Loading processed data from {path}")
    return pd.read_csv(path)


def get_features(df: pd.DataFrame):
    """Separate features and target."""
    target_col = "loan_status"
    X = df.drop(columns=[target_col, "loan_amnt"])  # drop loan_amnt to avoid multicollinearity with loan_percent_income
    y = df[target_col]
    return X, y


def train_and_eval(X, y):
    """Train Logistic Regression, evaluate, and log to MLflow."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        f"Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Train default rate: {y_train.mean():.3f} | Test default rate: {y_test.mean():.3f}"
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)


    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Test AUC-ROC: {auc:.4f}")
    logger.info(f"Test F1-Score: {f1:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # Feature importances (coefficients)
    coef_df = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
    logger.info("\nTop 5 feature coefficients:\n" + str(coef_df.head()))

    return model, scaler, auc, f1, coef_df, X_train, X_test, y_train, y_test


def main():
    df = load_processed(PROCESSED_DATA_PATH)
    X, y = get_features(df)

    with mlflow.start_run(run_name="logistic-regression-baseline"):
        model, scaler, auc, f1, coef_df, X_train, X_test, y_train, y_test = train_and_eval(X, y)

        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))

        # Log metrics
        mlflow.log_metric("test_auc_roc", auc)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("train_default_rate", y_train.mean())
        mlflow.log_metric("test_default_rate", y_test.mean())

        # Log model
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        # Log feature coefficients as artifact
        coef_df.to_csv("feature_coefficients.csv", header=True)
        mlflow.log_artifact("feature_coefficients.csv")

    logger.info("MLflow run complete. Start UI with: mlflow ui --backend-store-uri file:///C:/Users/dell/GitHub/smart-loan-recovery/mlruns")


if __name__ == "__main__":
    main()
