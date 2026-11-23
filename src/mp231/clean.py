import pandas as pd

from src.mp231.data import PARAMS


PREPROCESSING = PARAMS["preprocessing"]


def fix_spelling_variants(df):
    """Standardize categorical values."""
    df = df.copy()

    for column in PREPROCESSING["categorical_columns"]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
                .str.replace(".", "", regex=False)
            )

    if "atm_type" in df.columns:
        atm_map = {
            "bank": "bank_branch",
            "bankbranch": "bank_branch",
            "bank_branch": "bank_branch",
            "bank_branch_atm": "bank_branch",
            "ban": "bank_branch",
            "mall_atm": "mall",
            "mall": "mall",
            "mal": "mall",
            "standalone_atm": "standalone",
            "standalone": "standalone",
            "sta": "standalone",
        }
        df["atm_type"] = df["atm_type"].replace(atm_map)

    return df


def remove_leaking_columns(df):
    """Remove columns that reveal future information."""
    df = df.copy()
    columns = PREPROCESSING["leaking_columns"]
    return df.drop(columns=columns, errors="ignore")


def fix_sentinel_values(df):
    """Replace the sentinel value with missing data."""
    df = df.copy()

    column = PREPROCESSING["sentinel_column"]
    value = PREPROCESSING["sentinel_value"]

    if column in df.columns:
        df[column] = df[column].replace(value, float("nan"))

    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def fix_target_typos(df):
    """Correct unusually large target values."""
    df = df.copy()

    target = PREPROCESSING["target_column"]
    threshold = PREPROCESSING["target_typo_threshold"]

    if target in df.columns:
        mask = df[target] > threshold
        df.loc[mask, target] = df.loc[mask, target] / 10

    return df


def clean_train(df):
    """Apply all training-data cleaning steps."""
    df = fix_spelling_variants(df)
    df = remove_leaking_columns(df)
    df = fix_sentinel_values(df)
    df = remove_duplicates(df)
    df = fix_target_typos(df)

    return df


def clean_later(df):
    """Clean later data without changing its target values."""
    df = fix_spelling_variants(df)
    df = remove_leaking_columns(df)
    df = fix_sentinel_values(df)
    df = remove_duplicates(df)

    return df


def clean_holdout(df):
    """Clean holdout features."""
    df = fix_spelling_variants(df)
    df = fix_sentinel_values(df)
    df = remove_duplicates(df)

    return df


if __name__ == "__main__":
    from src.mp231.data import load_train, load_later, load_holdout

    train = clean_train(load_train())
    later = clean_later(load_later())
    holdout = clean_holdout(load_holdout())

    print("Cleaned train rows:", len(train))
    print("Cleaned later rows:", len(later))
    print("Cleaned holdout rows:", len(holdout))
    print("Leaking column removed:", "cash_reloaded_next_day" not in train.columns)
    print("Remaining sentinel values:",
          (train[PREPROCESSING["sentinel_column"]] == PREPROCESSING["sentinel_value"]).sum())
