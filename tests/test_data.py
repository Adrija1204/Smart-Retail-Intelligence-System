"""Tests for retail data ingestion."""

from pathlib import Path

import pandas as pd
import pytest

from smart_retail.data import DataValidationError, load_transactions


REQUIRED_STANDARD_COLUMNS = {
    "InvoiceNo",
    "UnitPrice",
    "CustomerID",
}


def test_load_transactions_standardizes_valid_rows(tmp_path: Path) -> None:
    """The loader converts dates and derives retail fields."""
    source = pd.DataFrame(
        {
            "InvoiceNo": ["10001", "C10002"],
            "StockCode": ["A", "B"],
            "Description": ["Product A", "Product B"],
            "Quantity": [2, -1],
            "InvoiceDate": ["2011-01-01", "2011-01-02"],
            "UnitPrice": [3.5, 4.0],
            "CustomerID": [12345, 12346],
            "Country": ["United Kingdom", "United Kingdom"],
        }
    )
    workbook = tmp_path / "retail.xlsx"
    source.to_excel(workbook, index=False)

    result = load_transactions(workbook)

    assert pd.api.types.is_datetime64_any_dtype(result["InvoiceDate"])
    assert result["is_cancelled"].tolist() == [False, True]
    assert result["line_revenue"].tolist() == [7.0, -4.0]


def test_load_transactions_rejects_missing_columns(tmp_path: Path) -> None:
    """The loader makes a schema mismatch explicit."""
    workbook = tmp_path / "bad_retail.xlsx"
    pd.DataFrame({"InvoiceNo": ["10001"]}).to_excel(workbook, index=False)

    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_transactions(workbook)


def test_load_transactions_standardizes_legacy_csv_headers(tmp_path: Path) -> None:
    """The loader translates the legacy CSV schema to the standard schema."""
    source = pd.DataFrame(
        {
            "Invoice": ["10001"],
            "StockCode": ["A"],
            "Description": ["Product A"],
            "Quantity": [2],
            "InvoiceDate": ["2011-01-01"],
            "Price": [3.5],
            "Customer ID": [12345],
            "Country": ["United Kingdom"],
        }
    )
    csv_path = tmp_path / "retail.csv"
    source.to_csv(csv_path, index=False)

    result = load_transactions(csv_path)

    assert REQUIRED_STANDARD_COLUMNS.issubset(result.columns)
