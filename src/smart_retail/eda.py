"""Reproducible exploratory analysis for retail transactions."""

import pandas as pd


SALES_COLUMNS = {
    "InvoiceNo",
    "Quantity",
    "UnitPrice",
    "is_cancelled",
    "line_revenue",
}


def _require_sales_columns(transactions: pd.DataFrame) -> None:
    """Raise an error when EDA input lacks required standardized fields."""
    missing_columns = sorted(SALES_COLUMNS.difference(transactions.columns))
    if missing_columns:
        raise ValueError(f"Missing EDA columns: {', '.join(missing_columns)}")


def valid_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return non-cancelled transactions with positive quantity and price."""
    _require_sales_columns(transactions)
    valid_rows = (
        ~transactions["is_cancelled"]
        & transactions["Quantity"].gt(0)
        & transactions["UnitPrice"].gt(0)
    )
    return transactions.loc[valid_rows].copy()


def build_kpis(transactions: pd.DataFrame) -> dict[str, float | int]:
    """Calculate core revenue and customer KPIs from valid sales."""
    sales = valid_sales(transactions)
    order_count = sales["InvoiceNo"].nunique()
    customer_count = sales["CustomerID"].nunique()
    revenue = sales["line_revenue"].sum()
    average_order_value = revenue / order_count if order_count else 0.0
    return {
        "valid_sales_rows": len(sales),
        "order_count": order_count,
        "customer_count": customer_count,
        "revenue": revenue,
        "average_order_value": average_order_value,
    }


def monthly_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid sales revenue by invoice month."""
    sales = valid_sales(transactions)
    summary = (
        sales.assign(month=sales["InvoiceDate"].dt.to_period("M"))
        .groupby("month", as_index=False)
        .agg(revenue=("line_revenue", "sum"))
    )
    summary["month"] = summary["month"].astype(str)
    return summary


def top_products(transactions: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return the highest-revenue products from valid sales."""
    sales = valid_sales(transactions)
    sales["Description"] = sales["Description"].fillna("Unknown product")
    return (
        sales.groupby("Description", as_index=False)
        .agg(revenue=("line_revenue", "sum"), units_sold=("Quantity", "sum"))
        .sort_values(["revenue", "Description"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )


def country_revenue(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return revenue and order count by country for valid sales."""
    sales = valid_sales(transactions)
    return (
        sales.groupby("Country", as_index=False)
        .agg(revenue=("line_revenue", "sum"), order_count=("InvoiceNo", "nunique"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
