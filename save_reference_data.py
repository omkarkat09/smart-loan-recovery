import pandas as pd
import numpy as np
import os

def generate_reference_data(output_path="data/reference.parquet", n_samples=1000):
    """Generates synthetic reference baseline data and saves to parquet."""
    np.random.seed(42)
    data = {
        'customer_id': [f"CUST-{i:05d}" for i in range(n_samples)],
        'loan_amount': np.random.uniform(1000, 50000, n_samples),
        'credit_score': np.random.normal(650, 50, n_samples).astype(int),
        'dti_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'days_overdue': np.random.exponential(15, n_samples).astype(int),
        'payment_history_6m': np.random.randint(0, 7, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"Reference baseline data saved to {output_path}")

if __name__ == "__main__":
    generate_reference_data()
