"""Tests for RFM features and customer segmentation."""

import pandas as pd

from smart_retail.segmentation import (
    build_rfm,
    evaluate_cluster_counts,
    fit_customer_segments,
    segment_profiles,
)


def _transactions() -> pd.DataFrame:
    """Return a compact standardized customer-sales table."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3", "4", "C5", "6"],
            "Quantity": [1, 2, 1, 5, -1, 1],
            "UnitPrice": [10.0, 10.0, 30.0, 5.0, 10.0, 8.0],
            "InvoiceDate": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-20",
                    "2024-01-25",
                    "2024-01-30",
                    "2024-01-31",
                    "2024-01-31",
                ]
            ),
            "CustomerID": pd.Series([1, 1, 2, 3, 1, 4], dtype="Int64"),
            "is_cancelled": [False, False, False, False, True, False],
            "line_revenue": [10.0, 20.0, 30.0, 25.0, -10.0, 8.0],
        }
    )


def test_build_rfm_calculates_customer_features() -> None:
    """RFM combines valid sales into one row per known customer."""
    result = build_rfm(_transactions(), reference_date="2024-02-01")
    customer_one = result.loc[result["CustomerID"] == 1].iloc[0]

    assert len(result) == 4
    assert customer_one["recency"] == 12
    assert customer_one["frequency"] == 2
    assert customer_one["monetary"] == 30.0


def test_evaluate_cluster_counts_returns_scores() -> None:
    """Feasible cluster counts produce one diagnostic row each."""
    rfm = build_rfm(_transactions(), reference_date="2024-02-01")
    result = evaluate_cluster_counts(rfm, cluster_counts=[2, 3])

    assert result["n_clusters"].tolist() == [2, 3]
    assert result["silhouette_score"].notna().all()


def test_fit_customer_segments_assigns_every_customer() -> None:
    """Every RFM row receives a numeric cluster and business segment."""
    rfm = build_rfm(_transactions(), reference_date="2024-02-01")
    customers, _, _ = fit_customer_segments(rfm, n_clusters=2)
    profiles = segment_profiles(customers)

    assert customers["segment"].notna().all()
    assert customers["cluster"].nunique() == 2
    assert profiles["customer_count"].sum() == len(rfm)

