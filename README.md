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

