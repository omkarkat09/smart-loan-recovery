# Smart Loan Recovery - Compliance & Ethics Checklist

As a financial machine learning application, Smart Loan Recovery must adhere to strict regulatory, legal, and ethical standards. This checklist serves as the operational baseline for maintaining compliance.

## 1. GDPR & CCPA: Right-to-Explanation
Under the General Data Protection Regulation (GDPR) and similar privacy frameworks, consumers subjected to automated decision-making have a "Right to Explanation" regarding how a decision (like risk tiering or collection intensity) was made.
- **Implementation:** We utilize **SHAP (SHapley Additive exPlanations)** in our Phase 3 architecture to calculate localized feature importance for every prediction.
- **Checklist:**
  - [ ] Ensure SHAP values are logged or easily retrievable for every prediction ID.
  - [ ] Customer Service representatives must have access to a dashboard that translates top SHAP features (e.g., "High Days Overdue", "Low Credit Score") into human-readable explanations when a borrower disputes a collection action.

## 2. ECOA / Fair Lending: Bias Checks
The Equal Credit Opportunity Act (ECOA) strictly prohibits discrimination on the basis of race, color, religion, national origin, sex, marital status, or age.
- **Implementation:** We utilize **Fairlearn** to evaluate model parity.
- **Checklist:**
  - [ ] Verify that protected attributes (e.g., age, gender, zip code proxies) are explicitly excluded from the `src/features/` pipeline.
  - [ ] Execute `run_phase3.py` (which includes the Fairlearn evaluation) prior to any major model architecture changes.
  - [ ] Ensure Demographic Parity Difference and Equalized Odds metrics remain within the mutually agreed legal tolerance bounds (< 5% variance across identified sensitive proxy groups).

## 3. FCRA: Data Retention & PII Policy
The Fair Credit Reporting Act (FCRA) regulates the collection, dissemination, and use of consumer information. Data minimization is a strict requirement.
- **Implementation:** Raw consumer financial data cannot be stored indefinitely.
- **Checklist:**
  - [ ] **90-Day Purge Rule:** Ensure automated cron jobs execute `DELETE` statements on the raw transactional and ingestion databases to purge Personally Identifiable Information (PII) after 90 days.
  - [ ] **Anonymization:** Feature datasets used in `data/reference.parquet` or stored in MLflow artifacts must have `customer_id` hashed or dropped.
  - [ ] **Access Control:** Ensure only authorized data science personnel can access the `data/raw/` directory.

## 4. Algorithmic Feedback Loops (Ouroboros Effect)
In Reinforcement Learning and Active Collection models, the model's actions influence the future data it trains on. If the RL Agent aggressively contacts "Medium Risk" borrowers, they may pay faster, causing the next model to view "Medium Risk" borrowers as "Low Risk", eventually leading to reduced contact and subsequent default spikes.
- **Implementation:** Active monitoring of intervention-outcome correlations.
- **Checklist:**
  - [ ] **Holdout Group:** Maintain a 5% random holdout control group (A/B testing variant where `recommended_channel = 'none'`) to measure baseline organic recovery rates without model intervention.
  - [ ] **Intervention Monitoring:** Actively monitor the `slr_collection_action_counter` in Grafana. If the RL agent collapses into selecting only one action (e.g., recommending 'High Intensity Phone Call' 100% of the time), trigger a manual retraining of the RL agent with higher exploration (Epsilon).
  - [ ] **Concept Drift:** Monitor `slr_drift_score`. If borrower behavioral changes shift the underlying feature distributions (due to our own collection strategies), the Stacking Ensemble must be retrained to learn the new baseline.
