"""RFM feature engineering and customer segmentation."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from smart_retail.eda import valid_sales


RFM_COLUMNS = ["recency", "frequency", "monetary"]
SEGMENT_LABELS = {
    2: ["High Value", "Lower Value"],
    3: ["Champions", "Potential Customers", "At Risk"],
    4: ["Champions", "Former High-Value", "Potential Customers", "At Risk"],
    5: [
        "Champions",
        "Loyal Customers",
        "Potential Customers",
        "At Risk",
        "Hibernating",
    ],
}


def _reference_date(sales: pd.DataFrame, reference_date: str | None) -> pd.Timestamp:
    """Return a normalized observation cutoff for recency calculation."""
    if reference_date is not None:
        return pd.Timestamp(reference_date).normalize()
    return sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def build_rfm(
    transactions: pd.DataFrame, reference_date: str | None = None
) -> pd.DataFrame:
    """Build recency, frequency, and monetary features for known customers."""
    sales = valid_sales(transactions).dropna(subset=["CustomerID"])
    if sales.empty:
        raise ValueError("Cannot build RFM features without valid customer sales.")

    cutoff = _reference_date(sales, reference_date)
    rfm = (
        sales.groupby("CustomerID", as_index=False)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("line_revenue", "sum"),
        )
        .assign(recency=lambda frame: (cutoff - frame["last_purchase"]).dt.days)
        .drop(columns="last_purchase")
        .loc[:, ["CustomerID", *RFM_COLUMNS]]
        .sort_values("CustomerID")
        .reset_index(drop=True)
    )
    return rfm


def _prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform and standardize RFM features for distance-based clustering."""
    if rfm.empty:
        raise ValueError("Cannot cluster an empty RFM table.")
    missing_columns = sorted(set(RFM_COLUMNS).difference(rfm.columns))
    if missing_columns:
        raise ValueError(f"Missing RFM columns: {', '.join(missing_columns)}")

    transformed = np.log1p(rfm[RFM_COLUMNS])
    scaler = StandardScaler()
    return scaler.fit_transform(transformed), scaler


def evaluate_cluster_counts(
    rfm: pd.DataFrame,
    cluster_counts: Sequence[int] = (2, 3, 4, 5, 6),
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate inertia and silhouette scores for feasible K-Means counts."""
    customer_count = len(rfm)
    candidates = sorted(set(cluster_counts))
    if not candidates or min(candidates) < 2 or max(candidates) >= customer_count:
        raise ValueError("Cluster counts must be between 2 and customer count - 1.")

    scaled_features, _ = _prepare_features(rfm)
    results = []
    for cluster_count in candidates:
        model = KMeans(n_clusters=cluster_count, n_init=20, random_state=random_state)
        labels = model.fit_predict(scaled_features)
        results.append(
            {
                "n_clusters": cluster_count,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(scaled_features, labels),
            }
        )
    return pd.DataFrame(results).sort_values("n_clusters").reset_index(drop=True)


def _cluster_labels(customers: pd.DataFrame) -> dict[int, str]:
    """Map cluster IDs to business labels using their combined RFM rank."""
    cluster_count = customers["cluster"].nunique()
    if cluster_count not in SEGMENT_LABELS:
        raise ValueError("Segment labels are available only for two to five clusters.")

    profiles = customers.groupby("cluster", as_index=False)[RFM_COLUMNS].mean()
    profiles["priority"] = (
        profiles["recency"].rank(method="first", ascending=True)
        + profiles["frequency"].rank(method="first", ascending=False)
        + profiles["monetary"].rank(method="first", ascending=False)
    )
    ranked_clusters = profiles.sort_values(["priority", "cluster"])["cluster"].tolist()
    return dict(zip(ranked_clusters, SEGMENT_LABELS[cluster_count], strict=True))


def fit_customer_segments(
    rfm: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    """Fit K-Means and attach ranked business segment labels to customers."""
    if n_clusters < 2 or n_clusters > 5 or n_clusters >= len(rfm):
        raise ValueError("n_clusters must be between 2 and min(5, customer count - 1).")

    scaled_features, scaler = _prepare_features(rfm)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    customers = rfm.copy()
    customers["cluster"] = model.fit_predict(scaled_features)
    customers["segment"] = customers["cluster"].map(_cluster_labels(customers))
    return customers, scaler, model


def segment_profiles(customers: pd.DataFrame) -> pd.DataFrame:
    """Summarize customer count and average RFM values by business segment."""
    required_columns = {"segment", *RFM_COLUMNS}
    missing_columns = sorted(required_columns.difference(customers.columns))
    if missing_columns:
        raise ValueError(f"Missing segment columns: {', '.join(missing_columns)}")

    return (
        customers.groupby("segment", as_index=False)
        .agg(
            customer_count=("CustomerID", "size"),
            average_recency=("recency", "mean"),
            average_frequency=("frequency", "mean"),
            average_monetary=("monetary", "mean"),
        )
        .sort_values("average_monetary", ascending=False)
        .reset_index(drop=True)
    )
