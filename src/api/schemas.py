"""Pydantic schemas for Smart Loan Recovery API."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional

class LoanFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique identifier for the customer", json_schema_extra={"example": "CUST-001"})
    loan_amount: float = Field(..., description="Total loan amount", json_schema_extra={"example": 5000.0})
    credit_score: int = Field(..., description="FICO credit score (300-850)", json_schema_extra={"example": 650})
    dti_ratio: float = Field(..., description="Debt-to-Income ratio (0.0 - 1.0)", json_schema_extra={"example": 0.45})
    days_overdue: int = Field(..., description="Number of days the loan is overdue", json_schema_extra={"example": 45})
    payment_history_6m: float = Field(..., description="Number of on-time payments in last 6 months", json_schema_extra={"example": 4})

    @field_validator('credit_score')
    @classmethod
    def validate_credit_score(cls, v):
        if v < 300 or v > 850:
            raise ValueError("Credit score must be between 300 and 850.")
        return v
        
    @field_validator('dti_ratio')
    @classmethod
    def validate_dti_ratio(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("DTI ratio must be between 0.0 and 1.0.")
        return v
        
    @field_validator('loan_amount')
    @classmethod
    def validate_loan_amount(cls, v):
        if v <= 0:
            raise ValueError("Loan amount must be positive.")
        return v
        
    @field_validator('days_overdue')
    @classmethod
    def validate_days_overdue(cls, v):
        if v < 0:
            raise ValueError("Days overdue must be non-negative.")
        return v

class PredictionResponse(BaseModel):
    customer_id: str
    default_probability: float
    risk_tier: str
    recovery_p25: Optional[float] = None
    recovery_p50: Optional[float] = None
    recovery_p75: Optional[float] = None
    recommended_channel: Optional[str] = None
    recommended_intensity: Optional[str] = None
    action_cost: Optional[float] = None
    expected_roi: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-001",
                "default_probability": 0.85,
                "risk_tier": "high",
                "recovery_p25": 100.0,
                "recovery_p50": 250.0,
                "recovery_p75": 500.0,
                "recommended_channel": "phone",
                "recommended_intensity": "high",
                "action_cost": 30.0,
                "expected_roi": 220.0
            }
        }
    }

class CollectionOptimizationRequest(BaseModel):
    customer_id: str
    risk_tier: str
