import pandas as pd
from fastapi.testclient import TestClient

from src.mp231.api import app
from src.mp231.clean import (
    remove_leaking_columns,
    fix_sentinel_values,
    remove_duplicates,
)

client = TestClient(app)


def test_leaking_column_removed():
    df = pd.DataFrame({
        "nearby_shops": [10],
        "cash_reloaded_next_day": [500]
    })
    result = remove_leaking_columns(df)
    assert "cash_reloaded_next_day" not in result.columns


def test_sentinel_value_replaced():
    df = pd.DataFrame({
        "distance_to_market_km": [2.0, 9999.0]
    })
    result = fix_sentinel_values(df)
    assert result["distance_to_market_km"].isna().sum() == 1


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
