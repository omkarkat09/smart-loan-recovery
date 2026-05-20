"""Unit tests for models module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.models.predict_model import predict

def test_predict_risk_tiers():
    """Test that predict() correctly assigns risk tiers and recommended actions."""
    # Mock model
    mock_model = MagicMock()
    # Mock predict_proba to return specific probabilities: 0.1, 0.45, 0.8
    # Format required by sklearn: array of shape (n_samples, n_classes)
    mock_model.predict_proba.return_value = np.array([
        [0.9, 0.1],  # low risk
        [0.55, 0.45], # medium risk
        [0.2, 0.8]   # high risk
    ])
    
    # Dummy features
    dummy_features = pd.DataFrame({"dummy": [1, 2, 3]})
    
    results = predict(dummy_features, model=mock_model)
    
    assert len(results) == 3
    assert results["default_probability"].iloc[0] == 0.1
    assert results["risk_tier"].iloc[0] == "low"
    assert "Standard Monitoring" in results["recommended_action"].iloc[0]
    
    assert results["default_probability"].iloc[1] == 0.45
    assert results["risk_tier"].iloc[1] == "medium"
    assert "Restructuring" in results["recommended_action"].iloc[1]
    
    assert results["default_probability"].iloc[2] == 0.8
    assert results["risk_tier"].iloc[2] == "high"
    assert "Aggressive" in results["recommended_action"].iloc[2]
