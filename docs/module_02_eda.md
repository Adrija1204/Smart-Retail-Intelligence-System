# Module 2: Exploratory data analysis

## Outcome

Produce reproducible retail metrics from valid sales only. A valid sale is a
non-cancelled row with positive quantity and positive unit price. This avoids
letting returns, credit notes, and malformed price records inflate revenue.

## Code guide (read before implementation)

`src/smart_retail/eda.py` will contain these lines and purposes:

1. The module docstring declares the EDA responsibility.
2. Pandas provides table operations and type annotations.
3. `SALES_COLUMNS` declares fields required for sales analysis.
4. `_require_sales_columns` identifies missing required fields.
5. It raises an explicit error when the input contract is broken.
6. `valid_sales` accepts standardized transactions.
7. It validates the input before filtering it.
8. The filter keeps non-cancelled, positive-quantity, positive-price rows.
9. `copy` prevents accidental mutation of upstream data.
10. The function returns clean sales for consistent downstream reuse.
11. `build_kpis` receives raw standardized transactions.
12. It calls `valid_sales` so every KPI uses the same business definition.
13. `order_count` counts distinct invoices, not line items.
14. `customer_count` counts known unique customers.
15. `revenue` sums derived line revenue.
16. `average_order_value` divides revenue by orders safely.
17. The return dictionary makes metric names explicit and serializable.
18. `monthly_revenue` reuses the valid-sales definition.
19. It groups each line by the invoice month.
20. It sums line revenue within each month.
21. It converts the period to a readable string for CSV and charts.
22. It returns a stable chronological table.
23. `top_products` reuses valid sales.
24. It fills missing descriptions so group-by does not drop stock codes.
25. It aggregates revenue and units by product description.
26. It sorts by revenue, then name for deterministic ties.
27. It limits output to the requested number of products.
28. `country_revenue` groups valid sales by country.
29. It sums revenue and counts unique invoices for each country.
30. It orders countries by revenue for business prioritization.

`scripts/run_eda.py` will contain these lines and purposes:

1. The module docstring describes the command-line purpose.
2. `argparse` handles explicit command options.
3. `Path` creates platform-safe folders and paths.
4. The project functions supply loading and EDA logic.
5. `main` is the executable workflow.
6. The parser accepts a source dataset path.
7. The parser accepts an output directory.
8. Parsed arguments become `Path` objects.
9. The output directory is created when absent.
10. The loader reads and standardizes the source dataset.
11. KPI output is written as one CSV row.
12–14. Monthly, product, and country analysis are each written as CSV files.
15. The print statement confirms where results were saved.
16–17. The main guard lets the file run as a script without side effects on import.

`tests/test_eda.py` will use a small transaction table to verify that cancelled,
zero-quantity, and zero-price rows are excluded; that revenue and AOV are
correct; and that the summary tables have the intended order.

## Run instructions

After implementation, run:

```powershell
python scripts/run_eda.py --input data/raw/online_retail.csv --output reports/eda
```

The command writes `kpis.csv`, `monthly_revenue.csv`, `top_products.csv`, and
`country_revenue.csv` to `reports/eda`.

## Interview questions

1. Why should cancellations be excluded from a gross-sales EDA view but retained in the raw table?
2. What is the difference between average order value and average line-item value?
3. How can a single outlier invoice distort revenue trends and clustering features?
4. Why is time-aware analysis important before building a future-purchase model?
