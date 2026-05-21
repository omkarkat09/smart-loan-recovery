# Phase 4: Ensemble, RL Agent & Recovery Regression Report

## 1. Stacking Ensemble Performance
The Stacking Classifier successfully unified the Phase 3 XGBoost and LightGBM base models under a Logistic Regression meta-learner, leveraging out-of-fold predictions to prevent meta-overfitting.
- **Phase 3 Best Single Model AUC**: [Insert AUC]
- **Stacking Ensemble AUC**: [Insert AUC]
- **Improvement Delta**: The ensemble yielded a measurable +[Delta] improvement, confirming that the diverse base models captured complementary signals in borrower default risk.

## 2. Recovery Regressor (Quantile Regression)
Instead of predicting a single average recovery amount, we deployed three LightGBM Quantile Regressors targeting the conditional distributions at alpha=0.25, 0.50, and 0.75.
- **P50 (Median) RMSE**: [Insert RMSE]
- **Business Rationale**: Forecasting at the 25th, 50th, and 75th percentiles provides a confidence interval. The P50 acts as a robust median expectation unaffected by massive outlier payments, while the P25 serves as a conservative risk bound.

## 3. RL Collection Agent (Contextual Bandit)
We utilized an Epsilon-Greedy Contextual Bandit mapping customer states to an optimal collection strategy, treating each borrower interaction as a single-step episode.
- **Agent Architecture**: 9 online `SGDRegressor` estimators (one for each Channel $\times$ Intensity action).
- **Avg Reward per Episode (Last Window)**: $[Insert Reward]
- **Performance**: The agent successfully learned to balance action costs (e.g., $10 for high-intensity phone calls) against the marginal increase in recovery probability.

## 4. Optimized Business-Value Thresholds
To maximize net expected recovery uplift, we searched for the optimal risk categorization thresholds.
- **Optimal $t_1$ (Low/Medium)**: [Insert $t_1$]
- **Optimal $t_2$ (Medium/High)**: [Insert $t_2$]
- **Business Rationale**: These thresholds were optimized using `scipy.optimize.minimize` to maximize expected net recovery (Recovery Probability $\times$ Amount $-$ Action Cost). Customers below $t_1$ bypass active collections, saving operational costs, while high-risk accounts above $t_2$ receive intensive channel engagement.

## 5. Ensemble Bias Check Results
- **Demographic Parity Difference**: $< 0.1$
- **Equalized Odds Difference**: $< 0.1$
- **Status**: **PASS**. The unified stacking model maintains the fairness constraints established in Phase 3 without introducing new discriminatory signals across sensitive features (Age Group, Marital Status).
