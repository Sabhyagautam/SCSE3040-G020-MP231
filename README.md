# SCSE3040 – MLOps Challenge-Based Assignment

## Project Details

- **Group ID:** G020
- **Problem ID:** MP231
- **Problem Title:** ATM Cash Demand Prediction – Guwahati

## Team Members

| Name | Enrollment Number |
|---|---|
| MOULIKGAUR | S24CSEU2132 |
| HarshiniSaharan | S24CSEU0737 |
| RishabhYadav | S24CSEU2228 |
| SABHYAGAUTAM | S24CSEU2087 |

## Project Description

This project focuses on identifying data quality issues in ATM cash demand data and building a reproducible MLOps workflow for ATM cash demand prediction.

## Repository Structure

```text
G020_MP231/
├── data/
│   └── raw/
├── reports/
│   └── findings.yaml
├── src/
│   └── mp231/
│       └── investigate.py
├── .dvc/
├── .gitignore
└── README.md
```

## Data Version Control

DVC is used to track the training and later datasets.

- `train.csv`
- `later.csv`

The holdout features file is tracked through Git.
## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data Cleaning

The data cleaning module handles the identified data quality issues, including:
- Data leakage
- Sentinel values
- Duplicate records
- Target typos
- Spelling variations

The cleaned datasets are prepared for model training.

## Model Training and MLflow Tracking

Run the training pipeline:

```bash
python -m src.mp231.train
```

MLflow is used to track model experiments, parameters, and evaluation metrics.

- **Experiment name:** `atm_cash_demand`
- **Registered model:** `atm_demand_model`
- **Champion alias:** `champion`
- **Tracking database:** `mlflow.db`

To open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` in your browser.

## Running Tests

Run the test suite:

```bash
python -m pytest -v
```

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- DVC
- FastAPI
- Pytest

