"""Data ingestion and validation for retail transactions."""

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}

COLUMN_ALIASES = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}


class DataValidationError(ValueError):
    """Raised when a source dataset violates the expected schema."""


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise an error when required source columns are missing."""
    missing_columns = sorted(set(required).difference(frame.columns))
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )


def _read_source(source_path: Path) -> pd.DataFrame:
    """Read a supported retail data file into a DataFrame."""
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source_path)
    raise ValueError(f"Unsupported dataset format: {source_path.suffix}")


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load a validated retail transaction table from an Excel workbook."""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Retail dataset not found: {source_path}")

    transactions = _read_source(source_path)
    transactions = transactions.rename(columns=COLUMN_ALIASES)
    _require_columns(transactions, REQUIRED_COLUMNS)

    transactions = transactions.copy()
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], errors="coerce"
    )
    transactions["CustomerID"] = pd.to_numeric(
        transactions["CustomerID"], errors="coerce"
    ).astype("Int64")
    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"], errors="coerce"
    )
    transactions["UnitPrice"] = pd.to_numeric(
        transactions["UnitPrice"], errors="coerce"
    )

    invalid_key_fields = transactions[
        ["InvoiceDate", "Quantity", "UnitPrice"]
    ].isna().any(axis=1)
    transactions = transactions.loc[~invalid_key_fields].copy()
    transactions["InvoiceNo"] = transactions["InvoiceNo"].astype(str)
    transactions["is_cancelled"] = transactions["InvoiceNo"].str.startswith("C")
    transactions["line_revenue"] = (
        transactions["Quantity"] * transactions["UnitPrice"]
    )
    return transactions
