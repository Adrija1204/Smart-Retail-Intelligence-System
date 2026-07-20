"""Tests for leakage-safe purchase propensity feature construction."""

import pandas as pd

from smart_retail.modeling import (
    build_purchase_dataset,
    temporal_train_test_split,
)


def _transactions() -> pd.DataFrame:
    """Return two customers with history before and after a March snapshot."""
    return pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3"],
            "Quantity": [1, 1, 1],
            "UnitPrice": [10.0, 20.0, 10.0],
            "InvoiceDate": pd.to_datetime(
                ["2024-01-01", "2024-02-10", "2024-03-10"]
            ),
            "CustomerID": pd.Series([1, 2, 1], dtype="Int64"),
            "is_cancelled": [False, False, False],
            "line_revenue": [10.0, 20.0, 10.0],
        }
    )


def test_build_purchase_dataset_creates_future_purchase_label() -> None:
    """A post-snapshot purchase becomes a label, not a feature."""
    dataset = build_purchase_dataset(
        _transactions(), [pd.Timestamp("2024-03-01")], history_days=90, horizon_days=30
    )

    customer_one = dataset.loc[dataset["CustomerID"] == 1].iloc[0]
    customer_two = dataset.loc[dataset["CustomerID"] == 2].iloc[0]
    assert customer_one["recency"] == 60
    assert customer_one["will_purchase"] == 1
    assert customer_two["will_purchase"] == 0


def test_temporal_split_reserves_latest_snapshot() -> None:
    """The test set contains dates later than every training date."""
    dataset = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
            "will_purchase": [0, 1, 0, 1],
        }
    )

    train, test = temporal_train_test_split(dataset, test_snapshots=1)

    assert train["snapshot_date"].max() < test["snapshot_date"].min()
    assert len(test) == 1

