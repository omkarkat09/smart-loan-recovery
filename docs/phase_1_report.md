# Phase 1 Report — Smart Loan Recovery: EDA

**Date:** 2026-04-27
**Status:** ✅ Complete
**Dataset:** `C:/Users/dell/Documents/GitHub/smart-loan-recovery/data/raw/credit_risk_dataset.csv`

---

## Data Sources

| File | Status | Records | Notes |
|---|---|---|---|
| `credit_risk_dataset.csv` | ✅ Active | 32,581 | Primary dataset — used here |
| `accepted_2007_to_2018Q4.csv` | 🔜 Deferred | ~2.6M | LendingClub accepted loans — too large for initial EDA; to be merged in later phase |
| `rejected_2007_to_2018Q4.csv` | 🔜 Deferred | — | Permission denied on read; to be resolved |
| `Data Dictionary.xls` | 🔜 Deferred | — | Column definitions for LendingClub data |

---

## 1. Data Quality Check

### Dataset Overview
| Property | Value |
|---|---|
| Total Records | 32,581 |
| Total Columns | 12 (11 predictors + 1 target) |
| Target Column | `loan_status` (0 = Non-Default, 1 = Default) |
| ID Column | None |

### Column Descriptions
| Column | Description | Dtype |
|---|---|---|
| `person_age` | Borrower age (years) | int64 |
| `person_income` | Annual income ($) | int64 |
| `person_home_ownership` | Home ownership status (RENT/OWN/MORTGAGE/OTHER) | string |
| `person_emp_length` | Employment length (years) | float64 |
| `loan_intent` | Purpose of loan (EDUCATION/MEDICAL/PERSONAL/etc.) | string |
| `loan_grade` | LendingClub loan grade (A–G) | string |
| `loan_amnt` | Loan amount ($) | int64 |
| `loan_int_rate` | Interest rate (%) | float64 |
| `loan_status` | **Target** — 0=Non-Default, 1=Default | int64 |
| `loan_percent_income` | Loan amount as % of annual income | float64 |
| `cb_person_default_on_file` | Previous default on file (Y/N) | string |
| `cb_person_cred_hist_length` | Length of credit history (years) | int64 |

### Missing Values
| Column | Missing | Percentage |
|---|---|---|
| `loan_int_rate` | 3,116 | 9.6% |
| `person_emp_length` | 895 | 2.7% |
| All other columns | 0 | 0% |

### Data Quality Issues
- **`loan_int_rate` has 9.6% missing values** — the most impactful column (highest correlation with target). Imputation strategy must be carefully chosen (median or model-based).
- **`person_emp_length` has 2.7% missing** — moderate concern; median imputation is acceptable.
- **`person_age` has outliers** — maximum age of 144 years is clearly erroneous (data entry error). Recommend filtering to age ≤ 100.
- No duplicate rows detected.

### Descriptive Statistics
| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| person_age | 27.73 | 6.35 | 20 | 26 | 144 ⚠️ |
| person_income | $69,902 | $63,117 | $6,000 | $55,000 | $6,000,000 |
| person_emp_length | 4.7 yrs | 4.0 | 0 | 4 | 41 |
| loan_amnt | $9,595 | $6,432 | $500 | $8,000 | $35,000 |
| loan_int_rate | 10.9% | 3.1 | 5.4% | 10.5% | 22.7% |
| loan_percent_income | 8.2% | 6.3 | 0.0% | 7.0% | 83.0% |
| cb_person_cred_hist_length | 5.8 yrs | 4.1 | 2 | 4 | 30 |

---

## 2. Target Distribution

| Class | Label | Count | Proportion |
|---|---|---|---|
| 0 | Non-Default | 25,473 | 78.2% |
| 1 | **Default** | 7,108 | **21.8%** |

**~4:1 class imbalance** (non-default to default ratio).

### Implications
- **Accuracy is not a reliable metric** — a naive model predicting all non-default would achieve 78.2% accuracy.
- **Primary metrics for evaluation:** AUC-ROC, F1-Score, Precision-Recall AUC
- **Recommended handling:** `class_weight='balanced'` in sklearn models; SMOTE/ADASYN on training folds only.
- The imbalance is **moderate** (not extreme) — well-tuned models with class weighting should perform well without heavy resampling.

---

## 3. Correlation Matrix

### Pearson Correlation with `loan_status` (sorted by absolute value)

| Rank | Feature | Correlation | Interpretation |
|---|---|---|---|
| 1 | `loan_int_rate` | **+0.38** | **Strongest predictor** — higher interest rate → higher default probability |
| 2 | `loan_percent_income` | **+0.32** | **Strong** — larger loan relative to income → higher risk |
| 3 | `cb_person_default_on_file` | **+0.27** *(enc.)* | Prior default on file is a strong risk flag |
| 4 | `loan_grade` | ordinal | A→G increases default; strongly correlated with int_rate |
| 5 | `person_income` | **−0.19** | Higher income → lower default risk |
| 6 | `cb_person_cred_hist_length` | **−0.07** | Longer credit history → slightly lower risk |
| 7 | `person_emp_length` | **−0.05** | Longer employment → slightly lower risk |
| 8 | `person_age` | **−0.03** | Minimal predictive power (but outliers may suppress this) |

### Key Correlation Insights
1. **`loan_int_rate` is the single most predictive feature** — higher-rate loans carry substantially higher default risk. This is both economically logical and the strongest numerical signal.
2. **`loan_percent_income` is a strong secondary predictor** — the debt burden relative to income is a well-known risk driver.
3. **`loan_grade` acts as a risk tier** — grade A loans default far less than grade G loans. Visual analysis (Section 5) shows a near-monotonic increase in default rate from grade A to G.
4. **`person_income` is negatively correlated** — higher earners are better risks, but the relationship is moderate.
5. **No serious multicollinearity** among top predictors — `loan_int_rate`, `loan_percent_income`, and `loan_grade` each provide independent signal.

### Note on `loan_grade`
`loan_grade` is ordinal (A > B > C > D > E > F > G) and should be encoded as integers 1–7 for modeling. It is highly correlated with `loan_int_rate` (r ≈ 0.85), so including both in linear models may cause multicollinearity — consider dropping one or using VIF screening.

---

## 4. Categorical Feature Distributions

| Feature | Categories | Dominant Category |
|---|---|---|
| `person_home_ownership` | RENT, OWN, MORTGAGE, OTHER | RENT (~50%) |
| `loan_intent` | EDUCATION, MEDICAL, PERSONAL, VENTURE, DEBTCONSOLIDATION, HOMEIMPROVEMENT | EDUCATION (~27%) |
| `loan_grade` | A, B, C, D, E, F, G | B (~34%) |
| `cb_person_default_on_file` | Y, N | N (~82%) |

- **`loan_intent`** is well-distributed across categories — each represents a distinct borrower motivation.
- **`cb_person_default_on_file = Y`** is a small but high-risk subgroup (~18%) — investigate separately.
- **Loan grades B and C** together account for ~60% of loans; A-grade loans are ~20%.

---

## 5. Key Visual Findings

*(See corresponding plots in `notebooks/`)*

| Plot | File | Key Finding |
|---|---|---|
| Target distribution | `02_target_dist.png` | 21.8% default rate — visible class imbalance |
| Numerical distributions | `03_num_dist.png` | Income & loan amounts are right-skewed; age is concentrated 22–35 |
| Categorical distributions | `04_cat_dist.png` | RENT dominates homeownership; EDUCATION most common loan intent |
| Correlation heatmap | `05_corr_matrix.png` | `loan_int_rate` and `loan_percent_income` dominate; income protective |
| Boxplots by default | `06_boxplots.png` | Defaulters have: higher int_rate, higher loan_percent_income, lower income |
| Default rate by grade | `07_grade_default.png` | Grade A ~3% default → Grade G ~80% default — near-perfect ordinal separation |

---

## 6. EDA Summary & Recommendations

### Summary
The `credit_risk_dataset.csv` contains 32,581 loan records with a **21.8% overall default rate**. The data is in good structural condition — no missing targets, no duplicate rows, only two columns with mild missingness. The main quality concerns are the 9.6% missing interest rates and age outliers. The strongest predictors of default are **interest rate**, **loan percent of income**, and **prior default history**.

### Recommended Preprocessing Steps (Phase 2)
1. **Filter** `person_age > 100` as erroneous
2. **Impute** `loan_int_rate` (consider modeling vs. median — 9.6% is non-trivial)
3. **Impute** `person_emp_length` with median
4. **Encode** `loan_grade` as ordinal integers (A=1 through G=7)
5. **Binary encode** `cb_person_default_on_file` (Y=1, N=0)
6. **One-hot encode** `loan_intent`, `person_home_ownership`
7. **Class balancing** — `class_weight='balanced'` or SMOTE on training splits

### Recommended Baseline Models (Phase 2)
- Logistic Regression (interpretable, good baseline)
- Random Forest / XGBoost (handles non-linearity and interactions)
- Evaluate with **AUC-ROC**, **F1**, **Precision-Recall curves**
- Stratified K-Fold cross-validation to preserve class distribution

---

*Artifacts: `notebooks/01_quality.png`, `02_target_dist.png`, `03_num_dist.png`, `04_cat_dist.png`, `05_corr_matrix.png`, `06_boxplots.png`, `07_grade_default.png`*  
*Generated from: `notebooks/01_eda_initial.ipynb`*  
*Dataset source: `C:/Users/dell/Documents/GitHub/smart-loan-recovery/data/raw/credit_risk_dataset.csv`*
