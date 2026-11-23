from pathlib import Path
import hashlib

import pandas as pd
import yaml


# Find the main project folder
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load project settings
with open(REPO_ROOT / "params.yaml", "r") as file:
    PARAMS = yaml.safe_load(file)

PATHS = PARAMS["paths"]


def load_train():
    """Load the original training dataset."""
    return pd.read_csv(REPO_ROOT / PATHS["train_raw"])


def load_later():
    """Load the later dataset."""
    return pd.read_csv(REPO_ROOT / PATHS["later_raw"])


def load_holdout():
    """Load the holdout dataset."""
    return pd.read_csv(REPO_ROOT / PATHS["holdout_raw"])


def sha256_of_train():
    """Calculate SHA-256 of the original, unchanged train.csv."""
    file_path = REPO_ROOT / PATHS["train_raw"]

    with open(file_path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


if __name__ == "__main__":
    train = load_train()
    later = load_later()
    holdout = load_holdout()

    print("Train rows:", len(train))
    print("Later rows:", len(later))
    print("Holdout rows:", len(holdout))
    print("Original train SHA-256:", sha256_of_train())
