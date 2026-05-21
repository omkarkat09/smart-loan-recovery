"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app, models

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_models():
    """Mock the models so we don't load MLflow during tests."""
    mock_stacker = MagicMock()
    mock_rec = MagicMock()
    mock_agent = MagicMock()
    
    models['stacker'] = mock_stacker
    models['recovery_models'] = {'p25': mock_rec, 'p50': mock_rec, 'p75': mock_rec}
    models['rl_agent'] = mock_agent
    models['status'] = 'ready'
    yield
    models.clear()

@patch('src.api.main.predict')
def test_predict_valid_input(mock_predict):
    """Test valid input returns 200 and expected response."""
    mock_predict.return_value = [{
        "customer_id": "CUST-001",
        "default_probability": 0.8,
        "risk_tier": "high",
        "recovery_p25": 100.0,
        "recovery_p50": 200.0,
        "recovery_p75": 300.0,
        "recommended_channel": "phone",
        "recommended_intensity": "high",
        "action_cost": 30.0,
        "expected_roi": 170.0
    }]
    
    payload = {
        "customer_id": "CUST-001",
        "loan_amount": 5000.0,
        "credit_score": 650,
        "dti_ratio": 0.45,
        "days_overdue": 45,
        "payment_history_6m": 4
    }
    
    response = client.post("/predict/default", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST-001"
    assert data["risk_tier"] == "high"

def test_predict_missing_field():
    """Test that missing required fields return 422."""
    payload = {
        "customer_id": "CUST-001",
        # missing loan_amount
        "credit_score": 650,
        "dti_ratio": 0.45,
        "days_overdue": 45,
        "payment_history_6m": 4
    }
    
    response = client.post("/predict/default", json=payload)
    assert response.status_code == 422
    assert "loan_amount" in response.text

def test_predict_out_of_range_credit_score():
    """Test that an out-of-range credit score returns 422."""
    payload = {
        "customer_id": "CUST-001",
        "loan_amount": 5000.0,
        "credit_score": 900, # Out of range (300-850)
        "dti_ratio": 0.45,
        "days_overdue": 45,
        "payment_history_6m": 4
    }
    
    response = client.post("/predict/default", json=payload)
    assert response.status_code == 422
    assert "Credit score must be between 300 and 850" in response.text
