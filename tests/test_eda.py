"""Tests for retail exploratory analysis."""

import pandas as pd

from smart_retail.eda import (
    build_kpis,
    country_revenue,
    monthly_revenue,
    top_products,
    valid_sales,
)


def _transactions() -> pd.DataFrame:
    """Return a small standardized transaction table for EDA tests."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "1", "C2", "3", "4"],
            "Description": ["Alpha", "Beta", "Alpha", "Gamma", "Beta"],
            "Quantity": [2, 1, -2, 0, 3],
            "InvoiceDate": pd.to_datetime(["2024-01-01"] * 5),
            "UnitPrice": [10.0, 5.0, 10.0, 4.0, 5.0],
            "CustomerID": pd.Series([10, 10, 10, 20, 30], dtype="Int64"),
            "Country": ["UK", "UK", "UK", "France", "France"],
            "is_cancelled": [False, False, True, False, False],
            "line_revenue": [20.0, 5.0, -20.0, 0.0, 15.0],
        }
    )


def test_valid_sales_excludes_returns_and_non_positive_values() -> None:
    """Only genuine positive sales remain in the EDA population."""
    result = valid_sales(_transactions())

    assert result["InvoiceNo"].tolist() == ["1", "1", "4"]


def test_build_kpis_uses_only_valid_sales() -> None:
    """Revenue, order count, and AOV follow the valid-sale definition."""
    result = build_kpis(_transactions())

    assert result == {
        "valid_sales_rows": 3,
        "order_count": 2,
        "customer_count": 2,
        "revenue": 40.0,
        "average_order_value": 20.0,
    }


def test_summary_tables_rank_by_revenue() -> None:
    """Product and country views return revenue-ranked results."""
    transactions = _transactions()

    assert top_products(transactions).iloc[0]["Description"] == "Alpha"
    assert country_revenue(transactions).iloc[0]["Country"] == "UK"


def test_monthly_revenue_has_readable_month_and_correct_total() -> None:
    """Monthly revenue is aggregated and ready for CSV or chart output."""
    result = monthly_revenue(_transactions())

    assert result.to_dict("records") == [{"month": "2024-01", "revenue": 40.0}]
