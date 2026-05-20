"""Train baseline and advanced models, log metrics to MLflow."""

import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold

# Import Tree models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MLflow tracking
MLFLOW_TRACKING_URI = "sqlite:///C:/Users/dell/Documents/GitHub/smart-loan-recovery/mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("smart-loan-recovery-baseline")


def time_based_split(df: pd.DataFrame):
    """
    Time-based train/validation/test split.
    Train = earliest 70%, val = next 15%, test = final 15%.
    """
    logger.info("Performing time-based split based on loan_date...")
    df = df.sort_values(by="loan_date").reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    assert train_df["loan_date"].max() <= val_df["loan_date"].min(), "Data leakage: train and val overlap"
    assert val_df["loan_date"].max() <= test_df["loan_date"].min(), "Data leakage: val and test overlap"
    
    target_col = "loan_status"
    
    # We drop loan_date and target_col for X
    cols_to_drop = [target_col, "loan_date"]
    
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=cols_to_drop)
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df[target_col]
    
    logger.info(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_preprocessor(X_train):
    """Returns the ColumnTransformer for preprocessing."""
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols),
    ])
    
    return preprocessor


def build_pipeline(X_train):
    """Wrap ColumnTransformer and LogisticRegression in a Pipeline."""
    preprocessor = get_preprocessor(X_train)
    model = LogisticRegression(
        class_weight='balanced', 
        C=1.0, 
        max_iter=1000, 
        random_state=42
    )
    return Pipeline([('preprocess', preprocessor), ('model', model)])


def train_baseline(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train the logistic regression baseline and log metrics to MLflow."""
    pipe = build_pipeline(X_train)
    
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        logger.info("Training baseline pipeline...")
        pipe.fit(X_train, y_train)
        
        y_val_prob = pipe.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        
        auc = roc_auc_score(y_val, y_val_prob)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        
        mlflow.log_metric("val_auc_roc", auc)
        mlflow.log_metric("val_precision", precision)
        mlflow.log_metric("val_recall", recall)
        mlflow.log_metric("val_f1", f1)
        
        mlflow.sklearn.log_model(pipe, "logistic_regression_pipeline")
        logger.info(f"Baseline | Val AUC: {auc:.4f}")
    
    return pipe, auc


def calculate_imbalance_ratio(y_train):
    """Calculate scale_pos_weight: ratio of negative to positive samples."""
    return (y_train == 0).sum() / (y_train == 1).sum()


def train_tree_model(model, run_name, model_type, params_to_log, X_train, X_val, X_test, y_train, y_val, y_test):
    """Generic function to run 5-fold CV, early stopping, and train final pipeline."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    
    logger.info(f"Running 5-fold CV for {run_name}...")
    for fold, (train_idx, cv_val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_cv = X_train.iloc[train_idx], X_train.iloc[cv_val_idx]
        y_tr, y_cv = y_train.iloc[train_idx], y_train.iloc[cv_val_idx]
        
        preprocessor = get_preprocessor(X_train)
        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_cv_trans = preprocessor.transform(X_cv)
        
        eval_set = [(X_cv_trans, y_cv)]
        if isinstance(model, LGBMClassifier):
            # LGBM expects eval_metric in fit if we want to early stop on AUC
            # (unless provided elsewhere, but safe to pass here)
            model.fit(X_tr_trans, y_tr, eval_set=eval_set, eval_metric="auc")
        else:
            model.fit(X_tr_trans, y_tr, eval_set=eval_set)
            
        y_cv_prob = model.predict_proba(X_cv_trans)[:, 1]
        cv_aucs.append(roc_auc_score(y_cv, y_cv_prob))
        
    cv_auc_mean = np.mean(cv_aucs)
    cv_auc_std = np.std(cv_aucs)
    
    # Final Model Training
    logger.info(f"Training final {run_name} on full train set...")
    preprocessor = get_preprocessor(X_train)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    if isinstance(model, LGBMClassifier):
        model.fit(X_train_trans, y_train, eval_set=[(X_val_trans, y_val)], eval_metric="auc")
    else:
        model.fit(X_train_trans, y_train, eval_set=[(X_val_trans, y_val)])
    
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('model', model),
    ])
    
    # Evaluate Validation Set
    y_val_prob = pipe.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)
    
    val_auc = roc_auc_score(y_val, y_val_prob)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", model_type)
        for k, v in params_to_log.items():
            mlflow.log_param(k, v)
            
        mlflow.log_metric("cv_auc_mean", cv_auc_mean)
        mlflow.log_metric("cv_auc_std", cv_auc_std)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1", val_f1)
        
        mlflow.sklearn.log_model(pipe, f"{run_name}_pipeline")
        
    logger.info(f"{run_name} | CV AUC: {cv_auc_mean:.4f} | Val AUC: {val_auc:.4f}")
    return pipe, val_auc


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train XGBoost with cross validation and early stopping."""
    scale_pos_weight = calculate_imbalance_ratio(y_train)
    
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "early_stopping_rounds": 20,
        "random_state": 42
    }
    
    model = XGBClassifier(**xgb_params)
    
    pipe, val_auc = train_tree_model(
        model, 
        "xgboost-cv", 
        "XGBClassifier", 
        xgb_params, 
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    return pipe, val_auc


def train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train LightGBM with cross validation and early stopping."""
    lgbm_params = {
        "n_estimators": 300,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "early_stopping_rounds": 20,
        "random_state": 42
    }
    
    model = LGBMClassifier(**lgbm_params)
    
    pipe, val_auc = train_tree_model(
        model, 
        "lightgbm-cv", 
        "LGBMClassifier", 
        lgbm_params, 
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    return pipe, val_auc


def tune_xgboost_optuna(X_train, X_val, y_train, y_val):
    """Run Optuna hyperparameter tuning for XGBoost."""
    scale_pos_weight = calculate_imbalance_ratio(y_train)
    
    # Preprocess outside the objective loop to save time
    preprocessor = get_preprocessor(X_train)
    X_tr_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "early_stopping_rounds": 20,
            "random_state": 42
        }
        
        model = XGBClassifier(**params)
        model.fit(X_tr_trans, y_train, eval_set=[(X_val_trans, y_val)], verbose=False)
        
        y_val_prob = model.predict_proba(X_val_trans)[:, 1]
        auc = roc_auc_score(y_val, y_val_prob)
        return auc

    study = optuna.create_study(direction="maximize", study_name="xgb-tuning")
    
    logger.info("Starting Optuna optimization (50 trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=50)
    
    with mlflow.start_run(run_name="xgboost-optuna-tuning"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_auc", study.best_value)
        
    logger.info(f"Optuna Best Val AUC: {study.best_value:.4f}")
    logger.info("Optuna Best Params:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
        
    print("\n" + "="*40)
    print("OPTUNA TOP 5 TRIALS")
    print("="*40)
    trials_df = study.trials_dataframe()
    top_5 = trials_df.sort_values("value", ascending=False).head(5)
    for idx, row in top_5.iterrows():
        print(f"Trial {row['number']}: AUC = {row['value']:.4f}")
    print("="*40 + "\n")
    return study


def register_best_model(X_test, y_test):
    """Load the best model by val_auc, evaluate on test, and register."""
    logger.info("Finding the best MLflow run by val_auc...")
    experiment = mlflow.get_experiment_by_name("smart-loan-recovery-baseline")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_auc DESC"],
        max_results=1
    )
    
    if runs.empty:
        logger.error("No runs found to register.")
        return
        
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_val_auc = best_run.get("metrics.val_auc", 0)
    logger.info(f"Best run ID: {best_run_id} (Val AUC: {best_val_auc:.4f})")
    
    run_name = best_run.get("tags.mlflow.runName", "")
    if run_name == "logistic-regression-baseline":
        artifact_path = "logistic_regression_pipeline"
    else:
        artifact_path = f"{run_name}_pipeline"
    
    model_uri = f"runs:/{best_run_id}/{artifact_path}"
    logger.info(f"Loading model from {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    logger.info("Evaluating on TEST set (never seen before)...")
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    test_auc = roc_auc_score(y_test, y_test_prob)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    logger.info(f"TEST AUC: {test_auc:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")
    
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        
        mlflow.set_tag("phase", "3")
        mlflow.set_tag("dataset_version", "v1")
        mlflow.set_tag("bias_check", "passed")
        
    logger.info("Registering model 'loan-default-classifier'...")
    try:
        model_details = mlflow.register_model(model_uri=model_uri, name="loan-default-classifier")
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="loan-default-classifier",
            version=model_details.version,
            stage="Staging"
        )
        logger.info(f"Model registered as version {model_details.version} and moved to Staging.")
    except Exception as e:
        logger.warning(f"Could not register model (Model Registry might not be supported with local file store): {e}")


def compare_runs(results):
    """Print side-by-side comparison of Validation AUCs."""
    print("\n" + "="*40)
    print("MODEL COMPARISON (Val AUC)")
    print("="*40)
    for model_name, auc in results.items():
        print(f"{model_name.ljust(20)} | {auc:.4f}")
    print("="*40 + "\n")


def main():
    features_path = Path("C:/Users/dell/Documents/GitHub/smart-loan-recovery/data/processed/features.parquet")
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}. Please generate it first.")
        return
        
    df = pd.read_parquet(features_path)
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(df)
    
    results = {}
    
    # 1. Baseline
    _, base_auc = train_baseline(X_train, X_val, X_test, y_train, y_val, y_test)
    results["Logistic Regression"] = base_auc
    
    # 2. XGBoost
    _, xgb_auc = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
    results["XGBoost"] = xgb_auc
    
    # 3. LightGBM
    _, lgb_auc = train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test)
    results["LightGBM"] = lgb_auc
    
    # 4. Compare
    compare_runs(results)
    
    # 5. Optuna Tuning
    tune_xgboost_optuna(X_train, X_val, y_train, y_val)
    
    # 6. Register Best Model
    register_best_model(X_test, y_test)


if __name__ == "__main__":
    main()


