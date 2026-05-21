"""Unit tests for the A/B testing framework."""

import pytest
import os
import sqlite3
import hashlib
from src.evaluation.ab_test import ABTestFramework

def test_deterministic_assignment():
    """Verify assign_variant deterministically hashes to control/treatment."""
    db_path = "tests/test_ab_db.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    ab = ABTestFramework(db_path=db_path)
    
    # Same user yields same result
    assert ab.assign_variant("user123") == ab.assign_variant("user123")
    
    # Exact calculation logic match
    h = int(hashlib.sha256(b"user123").hexdigest(), 16)
    expected = 'treatment' if h % 2 == 1 else 'control'
    assert ab.assign_variant("user123") == expected

def test_result_schema_and_compute():
    """Test db logging and computing proportions_ztest correctly."""
    db_path = "tests/test_ab_db_compute.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    ab = ABTestFramework(db_path=db_path)
    
    # 2 Control: 1 fail, 1 success (50%)
    ab.log_outcome("u1", "control", "email", 0.0, 1.0)
    ab.log_outcome("u2", "control", "sms", 100.0, 2.0)
    
    # 2 Treatment: 2 success (100%)
    ab.log_outcome("u3", "treatment", "phone", 500.0, 10.0)
    ab.log_outcome("u4", "treatment", "phone", 200.0, 10.0)
    
    results = ab.compute_results()
    
    assert results is not None
    assert results['rate_control'] == 0.5
    assert results['rate_treatment'] == 1.0
    assert 'p_value' in results
    assert isinstance(results['significant'], bool)
