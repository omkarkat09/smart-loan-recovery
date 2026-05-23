# Smart Loan Recovery

An end-to-end machine learning system that predicts loan default risk and recommends
optimal collection strategies to maximise recovery rates while minimising operational costs
and customer friction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Phase Roadmap](#phase-roadmap)
- [Success Metrics](#success-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

**Problem statement:**
> Given historical loan and repayment data, predict which borrowers are likely to default
> and recommend optimal collection actions (channel, timing, intensity) to maximise recovery rates.

**Four core models (built across phases):**

| Model | Type | Goal |
|---|---|---|
| Default Risk Prediction | Binary classification | Will this borrower default in the next 90 days? |
| Recovery Amount Prediction | Regression | How much can we expect to recover? |
| Optimal Action Recommendation | Reinforcement learning | Which channel and timing maximises collection? |
| Payment Behaviour Forecasting | Time series (LSTM) | Forecast payment patterns for next 6 months |

---

## Repository Structure

```
smart-loan-recovery/
├── .github/
│   └── workflows/
│       └── main.yml            # CI: lint + test on every push
├── docs/
│   ├── problem_statement.md    # Detailed problem definition
│   ├── architecture.md         # System design diagrams
│   └── phase_1_report.md       # EDA findings & feature ideas
├── data/
│   ├── raw/                    # Immutable source data (never edited)
│   └── processed/              # Cleaned & transformed data
├── notebooks/
│   ├── 01_eda_initial.ipynb    # Exploratory data analysis
│   ├── 02_feature_exploration.ipynb
│   └── 03_model_prototyping.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py     # Ingestion, cleaning, validation
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py   # Feature engineering pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py      # Training + MLflow logging
│   │   └── predict_model.py    # Inference helpers
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── api/
│       ├── main.py             # FastAPI app (Phase 8)
│       └── schemas.py
├── config/
│   ├── config.ini
│   └── settings.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── environment.yml             # Conda environment
├── requirements.txt            # Pip requirements
├── Dockerfile                  # Container definition (Phase 8)
├── .gitignore
├── README.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) **or** pip + venv
- Git

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/smart-loan-recovery.git
cd smart-loan-recovery
```

### 2 — Create the environment

**With Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate smart-loan-recovery
```

**With pip + venv:**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Get the dataset

Download a loan default dataset from Kaggle (see [Data Sources](#data-sources) below)
and place the raw CSV in `data/raw/`. The file is ignored by Git — never commit raw data.

```bash
# Example: using the Kaggle CLI
kaggle datasets download -d wordsforthewise/lending-club -p data/raw/ --unzip
```

### 4 — Run the EDA notebook

```bash
jupyter lab notebooks/01_eda_initial.ipynb
```

### 5 — Run tests

```bash
pytest tests/ -v
```

---

## Data Sources

| Dataset | Platform | Notes |
|---|---|---|
| [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) | Kaggle | Good starter — small, clean, binary target |
| [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) | Kaggle | Larger, richer features, more realistic |
| [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | Kaggle | Real-world, good for feature engineering practice |

> **Note:** Raw data files must never be committed to the repository.
> Add your downloaded files to `data/raw/` — the `.gitignore` covers them automatically.

---

## Development Workflow

### Branch strategy

```
main          ← stable, passing CI only
└── dev       ← integration branch
    ├── feat/eda-notebook
    ├── feat/feature-engineering
    └── fix/data-cleaning-nulls
```

Always work on a feature branch. Open a pull request into `dev`, not directly into `main`.

### Before committing

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Run tests
pytest tests/ -v
```

Pre-commit hooks automate this — see `.pre-commit-config.yaml` (to be added in Phase 2).

### MLflow experiment tracking

```bash
# Start the MLflow UI (from the project root)
mlflow ui

# Open in browser
open http://localhost:5000
```

All training runs log parameters and metrics to MLflow automatically.

---

## Phase Roadmap

| Week | Phase | Key Deliverable |
|---|---|---|
| 1–2 | **Phase 1** · Problem Definition & EDA | Data quality report, feature ideas, baseline |
| 3–4 | Phase 2 · Feature Engineering | Feature store, transformation pipelines |
| 5–6 | Phase 3 · Model Development | XGBoost/LightGBM, hyperparameter tuning |
| 7–8 | Phase 4 · Ensemble & Optimisation | Stacking, RL agent, SHAP interpretability |
| 9–10 | Phase 5 · Deployment | FastAPI, Docker, Kubernetes, CI/CD |
| 11–12 | Phase 6 · Monitoring & Handoff | Evidently AI, Grafana dashboards, docs |

---

## Success Metrics

| Metric | Target | How Measured |
|---|---|---|
| Default Prediction AUC-ROC | > 0.85 | Model evaluation on held-out test set |
| Recovery Rate Improvement | +20% | A/B testing vs. baseline strategy |
| Cost-to-Collect Reduction | −25% | Operational data comparison |
| Customer Churn Reduction | −15% | Post-collection surveys |

**Phase 1 MVP gate (Week 6):** AUC > 0.80 · Batch inference pipeline running · Basic API endpoint

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add temporal feature engineering"`
4. Push and open a pull request into `dev`

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
