"""Integration tests for the unified predict pipeline."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.models.predict_model import predict

def test_unified_prediction():
    """Test that predict() connects stacker, recovery models, and RL agent seamlessly."""
    mock_stacker = MagicMock()
    mock_stacker.predict_proba.return_value = np.array([[0.1, 0.8]]) # 0.8 => high risk
    
    mock_rec = MagicMock()
    mock_rec.predict.side_effect = [np.array([100.0]), np.array([200.0]), np.array([300.0])]
    mock_rec_models = {'p25': mock_rec, 'p50': mock_rec, 'p75': mock_rec}
    
    mock_agent = MagicMock()
    mock_agent.select_action.return_value = 8 # phone (2), high (2) -> cost 10 * 3 = 30
    
    df = pd.DataFrame(np.random.rand(1, 20))
    df['customer_id'] = 'CUST-1'
    
    results = predict(df, stacker=mock_stacker, recovery_models=mock_rec_models, rl_agent=mock_agent)
    res = results[0]
    
    assert res['customer_id'] == 'CUST-1'
    assert res['risk_tier'] == 'high'
    assert res['recovery_p50'] == 200.0
    assert res['recommended_channel'] == 'phone'
    assert res['recommended_intensity'] == 'high'
    assert res['action_cost'] == 30.0
    assert res['expected_roi'] == 200.0 - 30.0
