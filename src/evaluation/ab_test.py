"""A/B Testing Framework for Smart Loan Recovery."""

import hashlib
import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import norm

def proportions_ztest(count, nobs):
    p1 = count[0] / nobs[0]
    p2 = count[1] / nobs[1]
    p = (count[0] + count[1]) / (nobs[0] + nobs[1])
    se = np.sqrt(p * (1 - p) * (1/nobs[0] + 1/nobs[1]))
    if se == 0:
        return 0, 1.0
    z = (p1 - p2) / se
    pval = 2 * (1 - norm.cdf(np.abs(z)))
    return z, pval

import os

class ABTestFramework:
    """Manages deterministic variant assignments and outcome tracking for A/B testing."""
    
    def __init__(self, db_path="data/ab_results.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initializes the SQLite database schema."""
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS ab_results (
                customer_id TEXT PRIMARY KEY,
                variant TEXT,
                action_taken TEXT,
                payment_received REAL,
                cost REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        
    def assign_variant(self, customer_id):
        """Assign variant deterministically based on sha256 hash mod 2."""
        hash_hex = hashlib.sha256(str(customer_id).encode('utf-8')).hexdigest()
        hash_int = int(hash_hex, 16)
        return 'treatment' if hash_int % 2 == 1 else 'control'
        
    def log_outcome(self, customer_id, variant, action_taken, payment_received, cost):
        """Log the result of a collection action to the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO ab_results (customer_id, variant, action_taken, payment_received, cost)
            VALUES (?, ?, ?, ?, ?)
        ''', (str(customer_id), variant, str(action_taken), float(payment_received), float(cost)))
        conn.commit()
        conn.close()
        
    def compute_results(self):
        """Compute two-proportion z-test on recovery rate between control and treatment."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM ab_results", conn)
        conn.close()
        
        if df.empty:
            print("No data available.")
            return None
            
        df['success'] = (df['payment_received'] > 0).astype(int)
        
        control_df = df[df['variant'] == 'control']
        treatment_df = df[df['variant'] == 'treatment']
        
        if len(control_df) == 0 or len(treatment_df) == 0:
            print("Insufficient data for both variants.")
            return None
            
        n_control = len(control_df)
        success_control = control_df['success'].sum()
        
        n_treatment = len(treatment_df)
        success_treatment = treatment_df['success'].sum()
        
        rate_control = success_control / n_control if n_control > 0 else 0
        rate_treatment = success_treatment / n_treatment if n_treatment > 0 else 0
        
        count = np.array([success_treatment, success_control])
        nobs = np.array([n_treatment, n_control])
        
        stat, pval = proportions_ztest(count, nobs)
        
        uplift_pct = (rate_treatment - rate_control) * 100
        is_significant = bool(pval < 0.05)
        
        print("--- A/B Test Results ---")
        print(f"Control Rate:   {rate_control:.2%}")
        print(f"Treatment Rate: {rate_treatment:.2%}")
        print(f"Uplift:         {uplift_pct:+.2f}%")
        print(f"P-Value:        {pval:.4f}")
        print(f"Significant?    {'Yes' if is_significant else 'No'} (alpha=0.05)")
        
        return {
            'rate_control': rate_control,
            'rate_treatment': rate_treatment,
            'uplift_pct': uplift_pct,
            'p_value': float(pval),
            'significant': is_significant
        }

if __name__ == "__main__":
    pass
