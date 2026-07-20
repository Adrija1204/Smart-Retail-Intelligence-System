# Module 1: Data ingestion and validation

## Outcome

Load the UCI workbook into a consistent transaction table that downstream EDA
and machine-learning modules can trust.

## Code guide (read before implementation)

`src/smart_retail/data.py` will contain these lines and purposes:

1. The module docstring describes the module boundary.
2. `Path` gives platform-safe file paths.
3. `Iterable` describes a collection of column names.
4. Pandas supplies tables and type conversion.
5. `REQUIRED_COLUMNS` records the raw-data contract.
6. `DataValidationError` gives callers a meaningful, domain-specific failure.
7. `_require_columns` accepts an input table and required column collection.
8. It finds required names missing from the input.
9. It raises a clear exception when any are absent.
10. `load_transactions` exposes the public loader.
11. `path` accepts either a text path or `Path` instance.
12. The docstring states the returned table's guarantees.
13. `source_path` normalizes the input path.
14. The existence check prevents an obscure spreadsheet-reader error.
15. The raised error tells the user exactly which file is missing.
16. `read_excel` imports the workbook.
17. `_require_columns` enforces the source schema.
18. `copy` ensures this function owns its returned data.
19. `InvoiceDate` is converted to a real datetime type.
20. `errors="coerce"` turns bad dates into detectable missing values.
21. `CustomerID` is converted to pandas' nullable integer type.
22. `Quantity` is converted to a numeric value.
23. `UnitPrice` is converted to a numeric value.
24. A mask identifies rows whose key fields could not be parsed.
25. Invalid key rows are removed rather than silently used.
26. Invoice IDs are converted to strings so leading cancellation markers remain.
27. `is_cancelled` identifies invoices beginning with `C`.
28. `line_revenue` computes quantity times unit price.
29. `return` supplies the validated transaction table.

`tests/test_data.py` will contain these lines and purposes:

1. `Path` supports a temporary test file.
2. Pandas creates a small synthetic workbook.
3. Pytest asserts expected exceptions.
4. The application loader is the unit under test.
5. The custom exception is imported for precise assertion.
6. The test builds a valid two-row table.
7. The test writes it to a temporary Excel file.
8. The loader reads and validates that file.
9. The first assertion checks date conversion.
10. The second checks cancellation classification.
11. The third checks revenue calculation.
12. A second test creates a malformed source table.
13. It writes the malformed workbook.
14. The context manager expects the domain error.
15. Calling the loader triggers that error.

## Run instructions

Create a Python 3.11 virtual environment, install this package with its test
extras, place the workbook in `data/raw/online_retail.xlsx`, then use
`load_transactions("data/raw/online_retail.xlsx")` in the next module.

## CSV compatibility update

The downloaded file is the legacy Online Retail CSV format. Its `Invoice`,
`Price`, and `Customer ID` headers mean the same things as this project’s
`InvoiceNo`, `UnitPrice`, and `CustomerID` fields.

The update to `data.py` is explained line by line:

1. `COLUMN_ALIASES` records each legacy-to-standard header translation.
2. `_read_source` receives a normalized file path.
3. The suffix is normalized to lowercase so `.CSV` also works.
4. The CSV branch uses Pandas' CSV reader.
5. The Excel branch keeps workbook support.
6. The final branch raises a clear error for unsupported extensions.
7. The loader calls `_read_source` rather than assuming Excel.
8. `rename` applies the source-header translations before schema validation.

The accompanying test writes a CSV with legacy headers, loads it, and asserts
that all standard header names are present.

## Interview questions

1. Why does data validation belong at the ingestion boundary rather than in a model notebook?
2. What is the difference between coercing an invalid value and silently accepting it?
3. Why are cancellation flags retained instead of deleting cancelled invoices immediately?
4. Which data-quality metrics would you monitor in a scheduled production pipeline?
