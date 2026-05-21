import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import shap

from src.models.train_model import main as train_main, time_based_split
from src.models.predict_model import load_model, predict
from src.models.evaluate_model import check_bias

def run_all():
    print("=== 1. Training Models & Tuning ===")
    train_main()
    
    print("\n=== 2. Loading Best Model from Registry ===")
    model_pipeline = load_model()
    
    # Load test data
    features_path = "C:/Users/dell/Documents/GitHub/smart-loan-recovery/data/processed/features.parquet"
    df = pd.read_parquet(features_path)
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(df)
    
    print("\n=== 3. Bias Check ===")
    if 'age_group' in X_test.columns and 'marital_status' in X_test.columns:
        sensitive_df = X_test[['age_group', 'marital_status']]
        y_test_prob = model_pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        check_bias(y_test, y_test_pred, sensitive_df)
    else:
        print("Sensitive features missing. Skipping bias check.")
        
    print("\n=== 4. SHAP Analysis ===")
    # Extract model and preprocessor
    try:
        model = model_pipeline.named_steps['model']
        preprocessor = model_pipeline.named_steps['preprocess']
        
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        
        X_train_trans = pd.DataFrame(X_train_trans, columns=feature_names)
        X_test_trans = pd.DataFrame(X_test_trans, columns=feature_names)
        
        explainer = shap.TreeExplainer(model)
        # using test data for shap to be fast
        shap_values = explainer.shap_values(X_test_trans)
        
        shap.summary_plot(shap_values, X_test_trans, show=False)
        out_dir = "C:/Users/dell/Documents/GitHub/smart-loan-recovery/docs/model_evaluation"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/shap_summary_beeswarm.png", bbox_inches='tight')
        plt.close()
        print("SHAP plots generated and saved.")
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
    
    print("\n=== All Phase 3 Checks Completed ===")

if __name__ == "__main__":
    run_all()
