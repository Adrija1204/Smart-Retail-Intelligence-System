"""Export reproducible retail EDA tables from a source dataset."""

import argparse
from pathlib import Path

import pandas as pd

from smart_retail.data import load_transactions
from smart_retail.eda import build_kpis, country_revenue, monthly_revenue, top_products


def main() -> None:
    """Load retail data and save core EDA outputs as CSV files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source CSV or Excel file.")
    parser.add_argument("--output", required=True, help="Directory for EDA CSV outputs.")
    arguments = parser.parse_args()

    output_directory = Path(arguments.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    transactions = load_transactions(arguments.input)

    pd.DataFrame([build_kpis(transactions)]).to_csv(
        output_directory / "kpis.csv", index=False
    )
    monthly_revenue(transactions).to_csv(
        output_directory / "monthly_revenue.csv", index=False
    )
    top_products(transactions).to_csv(
        output_directory / "top_products.csv", index=False
    )
    country_revenue(transactions).to_csv(
        output_directory / "country_revenue.csv", index=False
    )
    print(f"EDA outputs written to: {output_directory}")


if __name__ == "__main__":
    main()

