"""Time-aware customer purchase propensity modeling."""

from collections.abc import Sequence

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from smart_retail.eda import valid_sales


FEATURE_COLUMNS = ["recency", "frequency", "monetary", "average_order_value"]
TARGET_COLUMN = "will_purchase"


def make_snapshot_dates(
    transactions: pd.DataFrame,
    history_days: int = 365,
    horizon_days: int = 30,
) -> list[pd.Timestamp]:
    """Return monthly snapshots with enough history and a complete horizon."""
    sales = valid_sales(transactions)
    start = sales["InvoiceDate"].min().normalize() + pd.Timedelta(days=history_days)
    end = sales["InvoiceDate"].max().normalize() - pd.Timedelta(days=horizon_days)
    snapshots = list(pd.date_range(start=start, end=end, freq="MS"))
    if len(snapshots) < 4:
        raise ValueError(
            "Dataset does not contain enough history for temporal modeling."
        )
    return snapshots


def _customer_features(
    history: pd.DataFrame, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """Create leakage-free customer features from a historical transaction view."""
    order_values = (
        history.groupby(["CustomerID", "InvoiceNo"], as_index=False)["line_revenue"]
        .sum()
        .rename(columns={"line_revenue": "order_value"})
    )
    customers = (
        history.groupby("CustomerID", as_index=False)
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("line_revenue", "sum"),
        )
        .merge(
            order_values.groupby("CustomerID", as_index=False)["order_value"]
            .mean()
            .rename(columns={"order_value": "average_order_value"}),
            on="CustomerID",
            how="left",
        )
    )
    customers["recency"] = (snapshot_date - customers["last_purchase"]).dt.days
    return customers.loc[:, ["CustomerID", *FEATURE_COLUMNS]]


def build_purchase_dataset(
    transactions: pd.DataFrame,
    snapshot_dates: Sequence[pd.Timestamp],
    history_days: int = 365,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Build a leakage-free supervised dataset from transaction snapshots."""
    sales = valid_sales(transactions).dropna(subset=["CustomerID"])
    snapshots = sorted(pd.Timestamp(date).normalize() for date in snapshot_dates)
    if not snapshots:
        raise ValueError("At least one snapshot date is required.")

    datasets = []
    for snapshot_date in snapshots:
        history_start = snapshot_date - pd.Timedelta(days=history_days)
        history = sales.loc[
            (sales["InvoiceDate"] >= history_start)
            & (sales["InvoiceDate"] < snapshot_date)
        ]
        if history.empty:
            continue
        customers = _customer_features(history, snapshot_date)
        future = sales.loc[
            (sales["InvoiceDate"] >= snapshot_date)
            & (sales["InvoiceDate"] < snapshot_date + pd.Timedelta(days=horizon_days))
        ]
        future_buyers = set(future["CustomerID"].dropna().unique())
        customers[TARGET_COLUMN] = customers["CustomerID"].isin(
            future_buyers
        ).astype(int)
        customers["snapshot_date"] = snapshot_date
        datasets.append(customers)

    if not datasets:
        raise ValueError("No customer snapshots could be built from the dataset.")
    return pd.concat(datasets, ignore_index=True)


def temporal_train_test_split(
    dataset: pd.DataFrame, test_snapshots: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a supervised dataset so evaluation is chronologically later."""
    dates = sorted(dataset["snapshot_date"].unique())
    if len(dates) <= test_snapshots:
        raise ValueError("More snapshot dates are required for the requested holdout.")
    test_dates = set(dates[-test_snapshots:])
    train = dataset.loc[~dataset["snapshot_date"].isin(test_dates)].copy()
    test = dataset.loc[dataset["snapshot_date"].isin(test_dates)].copy()
    return train, test


def build_models(
    random_state: int = 42,
) -> dict[str, Pipeline | RandomForestClassifier]:
    """Return a baseline and a nonlinear candidate for propensity prediction."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def evaluate_classifier(
    model: Pipeline | RandomForestClassifier, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Calculate ranking and threshold metrics for a fitted classifier."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }


def train_and_select_model(
    dataset: pd.DataFrame, random_state: int = 42
) -> tuple[Pipeline | RandomForestClassifier, str, pd.DataFrame, pd.DataFrame]:
    """Train candidates on past snapshots and select by future average precision."""
    train, test = temporal_train_test_split(dataset)
    evaluations = []
    fitted_models = {}
    for name, model in build_models(random_state).items():
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
        fitted_models[name] = model
        evaluations.append(
            {
                "model": name,
                **evaluate_classifier(
                    model, test[FEATURE_COLUMNS], test[TARGET_COLUMN]
                ),
            }
        )
    evaluation = pd.DataFrame(evaluations).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    selected_name = evaluation.iloc[0]["model"]
    return fitted_models[selected_name], selected_name, evaluation, test


def score_customers(
    model: Pipeline | RandomForestClassifier, customer_features: pd.DataFrame
) -> pd.DataFrame:
    """Attach future-purchase probabilities to one customer feature table."""
    scored = customer_features.copy()
    scored["purchase_probability"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    return scored.sort_values("purchase_probability", ascending=False).reset_index(
        drop=True
    )
