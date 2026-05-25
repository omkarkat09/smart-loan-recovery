"""Unit tests for models module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.models.predict_model import predict

def test_predict_risk_tiers():
    """Test that predict() correctly assigns risk tiers and recommended actions."""
    mock_stacker = MagicMock()
    mock_stacker.predict_proba.side_effect = [
        np.array([[0.9, 0.1]]),   # low risk (0.1)
        np.array([[0.55, 0.45]]), # medium risk (0.45)
        np.array([[0.2, 0.8]])    # high risk (0.8)
    ]
    
    mock_rec = MagicMock()
    mock_rec.predict.return_value = np.array([100.0])
    mock_rec_models = {'p25': mock_rec, 'p50': mock_rec, 'p75': mock_rec}
    
    mock_agent = MagicMock()
    mock_agent.select_action.return_value = 0 # email (0), low (0) -> cost 1 * 1 = 1
    
    # Dummy features (3 rows)
    dummy_features = pd.DataFrame(np.random.rand(3, 20))
    
    results = predict(dummy_features, stacker=mock_stacker, recovery_models=mock_rec_models, rl_agent=mock_agent)
    
    assert len(results) == 3
    
    # Low risk
    assert results[0]["default_probability"] == 0.1
    assert results[0]["risk_tier"] == "low"
    assert results[0]["recommended_channel"] == "none"
    assert results[0]["recommended_intensity"] == "none"
    
    # Medium risk
    assert results[1]["default_probability"] == 0.45
    assert results[1]["risk_tier"] == "medium"
    assert results[1]["recommended_channel"] == "email"
    assert results[1]["recommended_intensity"] == "low"
    
    # High risk
    assert results[2]["default_probability"] == 0.8
    assert results[2]["risk_tier"] == "high"
    assert results[2]["recommended_channel"] == "email"
    assert results[2]["recommended_intensity"] == "low"
