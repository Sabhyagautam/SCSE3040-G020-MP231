import pandas as pd

df = pd.read_csv("data/raw/train.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("\n1. LEAKING COLUMN")
for col in df.columns:
    if col not in holdout.columns:
        if col != "withdrawal_thousand_rs":
            corr = df[col].corr(df["withdrawal_thousand_rs"]) if pd.api.types.is_numeric_dtype(df[col]) else None
            print(col, "Correlation:", corr)

print("\n2. UNIT MIXUP: distance_to_market_km by area")
print(df.groupby("area")["distance_to_market_km"].agg(["min", "mean", "max"]))

print("\n3. SENTINEL VALUES")
print(df["distance_to_market_km"].value_counts().head(15))

print("\n4. DUPLICATE ROWS")
print("Extra duplicate rows:", df.duplicated().sum())

print("\n5. STUCK SENSOR: longest consecutive identical readings")
df = df.sort_values("date")
for col in df.select_dtypes(include="number").columns:
    if col in ["record_id"]:
        continue
    groups = df[col].ne(df[col].shift()).cumsum()
    runs = df.groupby(groups).agg(
        value=(col, "first"),
        count=(col, "size"),
        start=("date", "first"),
        end=("date", "last")
    )
    print(f"\n{col}:")
    print(runs[runs["count"] >= 10].sort_values("count", ascending=False).head(5))

print("\n6. TARGET TYPOS")
target = "withdrawal_thousand_rs"
print(df[df[target] > 2000][["record_id", "date", target]].to_string(index=False))

print("\n7. SPELLING VARIANTS")
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())
