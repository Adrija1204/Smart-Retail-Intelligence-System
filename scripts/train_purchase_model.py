"""Train and export a time-aware retail purchase propensity model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from smart_retail.data import load_transactions
from smart_retail.explainability import explain_logistic_pipeline, explain_random_forest
from smart_retail.modeling import (
    FEATURE_COLUMNS,
    build_purchase_dataset,
    make_snapshot_dates,
    score_customers,
    train_and_select_model,
)


def main() -> None:
    """Train, evaluate, explain, and export purchase propensity outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to source retail data.")
    parser.add_argument("--output", required=True, help="Directory for model reports.")
    parser.add_argument(
        "--model-path", required=True, help="Path for the Joblib model."
    )
    arguments = parser.parse_args()

    output_directory = Path(arguments.output)
    model_path = Path(arguments.model_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(arguments.input)
    snapshots = make_snapshot_dates(transactions)
    dataset = build_purchase_dataset(transactions, snapshots)
    model, model_name, evaluation, _ = train_and_select_model(dataset)

    current_snapshot = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(
        days=1
    )
    current_features = build_purchase_dataset(transactions, [current_snapshot]).drop(
        columns=["will_purchase", "snapshot_date"]
    )
    scored_customers = score_customers(model, current_features)

    evaluation.to_csv(output_directory / "model_evaluation.csv", index=False)
    dataset.to_csv(output_directory / "training_snapshots.csv", index=False)
    scored_customers.to_csv(output_directory / "customer_propensity.csv", index=False)
    joblib.dump(
        {"model": model, "model_name": model_name, "feature_columns": FEATURE_COLUMNS},
        model_path,
    )

    if isinstance(model, RandomForestClassifier):
        importance, contributions = explain_random_forest(
            model, scored_customers[FEATURE_COLUMNS]
        )
    else:
        importance, contributions = explain_logistic_pipeline(
            model, scored_customers[FEATURE_COLUMNS]
        )
    importance.to_csv(output_directory / "shap_global_importance.csv", index=False)
    contributions.to_csv(output_directory / "shap_local_contributions.csv", index=False)

    print(f"Selected model: {model_name}")
    print(f"Model reports written to: {output_directory}")
    print(f"Model bundle written to: {model_path}")


if __name__ == "__main__":
    main()
