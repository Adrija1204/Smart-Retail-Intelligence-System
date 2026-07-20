"""Export RFM customer segments and clustering diagnostics."""

import argparse
from pathlib import Path

from smart_retail.data import load_transactions
from smart_retail.segmentation import (
    build_rfm,
    evaluate_cluster_counts,
    fit_customer_segments,
    segment_profiles,
)


def main() -> None:
    """Build, evaluate, and export RFM customer segmentation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to retail source data.")
    parser.add_argument(
        "--output", required=True, help="Directory for segmentation outputs."
    )
    parser.add_argument(
        "--clusters", type=int, default=4, help="K-Means cluster count."
    )
    arguments = parser.parse_args()

    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    rfm = build_rfm(load_transactions(arguments.input))
    evaluation = evaluate_cluster_counts(rfm)
    customers, _, _ = fit_customer_segments(rfm, n_clusters=arguments.clusters)

    rfm.to_csv(output_directory / "rfm_features.csv", index=False)
    evaluation.to_csv(output_directory / "cluster_evaluation.csv", index=False)
    customers.to_csv(output_directory / "customer_segments.csv", index=False)
    segment_profiles(customers).to_csv(
        output_directory / "segment_profiles.csv", index=False
    )
    print(f"Segmentation outputs written to: {output_directory}")


if __name__ == "__main__":
    main()
