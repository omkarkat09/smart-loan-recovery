"""Unified inference pipeline chaining stacking, recovery, and RL models."""

import pandas as pd
import numpy as np
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///C:/Users/dell/Documents/GitHub/smart-loan-recovery/mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_all_models():
    """Load all Phase 4 models."""
    logger.info("Loading Stacking Ensemble...")
    stacker = mlflow.sklearn.load_model("models:/loan-default-stacker/Staging")
    
    logger.info("Loading Recovery Regressors...")
    recovery_models = {
        'p25': mlflow.sklearn.load_model("models:/loan-recovery-regressor-p25/Staging"),
        'p50': mlflow.sklearn.load_model("models:/loan-recovery-regressor-p50/Staging"),
        'p75': mlflow.sklearn.load_model("models:/loan-recovery-regressor-p75/Staging")
    }
    
    logger.info("Loading RL Bandit Agent...")
    rl_agent = mlflow.pyfunc.load_model("models:/collection-rl-agent/Staging") 
    return stacker, recovery_models, rl_agent
    
def predict(features_df, stacker=None, recovery_models=None, rl_agent=None, t1=0.3, t2=0.6):
    """
    1. Run stacking ensemble to get default_probability and risk_tier.
    2. If not 'low', run recovery model for p25, p50, p75.
    3. Run bandit agent for channel and intensity.
    4. Calculate expected ROI.
    """
    if stacker is None or recovery_models is None or rl_agent is None:
        stacker, recovery_models, rl_agent = load_all_models()
        
    results = []
    
    for idx, row in features_df.iterrows():
        row_df = pd.DataFrame([row])
        prob = float(stacker.predict_proba(row_df)[0, 1])
        
        risk_tier = 'low'
        if prob >= t2:
            risk_tier = 'high'
        elif prob >= t1:
            risk_tier = 'medium'
            
        res = {
            'customer_id': row.get('customer_id', idx),
            'default_probability': prob,
            'risk_tier': risk_tier
        }
        
        channels = {0: 'email', 1: 'sms', 2: 'phone'}
        intensities = {0: 'low', 1: 'medium', 2: 'high'}
        costs = {0: 1, 1: 2, 2: 10} 
        
        if risk_tier != 'low':
            rec_features = row_df.copy()
            rec_features['default_probability'] = prob
            
            p25 = float(recovery_models['p25'].predict(rec_features)[0])
            p50 = float(recovery_models['p50'].predict(rec_features)[0])
            p75 = float(recovery_models['p75'].predict(rec_features)[0])
            
            res['recovery_p25'] = p25
            res['recovery_p50'] = p50
            res['recovery_p75'] = p75
            
            state_vals = row_df.select_dtypes(include=[np.number]).values[0][:20]
            if len(state_vals) < 20:
                state_vals = np.pad(state_vals, (0, 20 - len(state_vals)), 'constant')
            else:
                state_vals = state_vals[:20]
                
            if hasattr(rl_agent, 'select_action'):
                action = rl_agent.select_action(state_vals)
            else:
                action = int(rl_agent.predict(state_vals.reshape(1, -1))[0])
                
            channel_idx = action // 3
            intensity_idx = action % 3
            action_cost = float(costs[channel_idx] * (intensity_idx + 1))
            
            res['recommended_channel'] = channels[channel_idx]
            res['recommended_intensity'] = intensities[intensity_idx]
            res['action_cost'] = action_cost
            res['expected_roi'] = float(p50 - action_cost)
            
        else:
            res['recovery_p25'] = None
            res['recovery_p50'] = None
            res['recovery_p75'] = None
            res['recommended_channel'] = 'none'
            res['recommended_intensity'] = 'none'
            res['action_cost'] = 0.0
            res['expected_roi'] = 0.0
            
        results.append(res)
        
    return results

if __name__ == "__main__":
    pass
