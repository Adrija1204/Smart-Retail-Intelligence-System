# Module 3: RFM customer segmentation

## Outcome

Turn transaction rows into one RFM record per identified customer, then group
customers using K-Means. RFM means:

- **Recency:** days since the customer's most recent completed purchase;
- **Frequency:** distinct completed invoices; and
- **Monetary:** completed-sales revenue.

The reference date is one day after the dataset's most recent transaction, so
the most recent customer has a recency of one day instead of zero.

## Code guide (read before implementation)

`src/smart_retail/segmentation.py` will contain these lines and purposes:

1. The module docstring defines the segmentation boundary.
2–4. NumPy, Pandas, and scikit-learn provide numerical, tabular, scaling, and clustering tools.
5. `valid_sales` reuses the agreed sales-quality definition from EDA.
6. `RFM_COLUMNS` records the three features used by the model.
7. `_reference_date` accepts the sales table and an optional explicit date.
8. An explicit date is normalized for reproducible backtests.
9. Otherwise, the latest invoice date plus one day becomes the observation cutoff.
10. `build_rfm` accepts raw standardized transactions and an optional cutoff.
11. It keeps valid sales with a known customer ID.
12. It rejects an empty customer population rather than fitting a meaningless model.
13. It obtains the observation cutoff.
14. It groups rows by customer ID.
15. Recency is measured from each customer's latest invoice date.
16. Frequency counts distinct invoices, avoiding line-item inflation.
17. Monetary sums completed line revenue.
18. The result is sorted by customer ID for stable exports.
19. `_prepare_features` selects RFM fields.
20. `log1p` reduces the influence of heavily skewed frequency and spend values.
21. `StandardScaler` puts all three transformed features onto comparable scales.
22. It returns both scaled values and the fitted scaler.
23. `evaluate_cluster_counts` accepts RFM data and candidate counts.
24. It validates that every candidate is feasible for the customer count.
25. It scales data once, then fits K-Means for each candidate.
26. Inertia measures within-cluster compactness.
27. Silhouette score measures separation relative to compactness.
28. A sorted table makes cluster-count selection auditable.
29. `fit_customer_segments` accepts RFM data, a selected count, and a seed.
30. It validates the selected count.
31. It scales features and fits deterministic K-Means with multiple starts.
32. It attaches numeric cluster IDs to each customer.
33. `_cluster_labels` ranks clusters by a combined RFM score.
34. Lower recency and higher frequency and monetary values rank better.
35. Predefined labels translate the rank to an action-friendly segment name.
36. The mapping is attached to customer rows.
37. The function returns customer segments, fitted scaler, and fitted model.
38. `segment_profiles` aggregates each segment's customer count and mean RFM values.
39. It sorts profiles by monetary value so stakeholders can prioritize them.

`scripts/run_segmentation.py` will load the source data, build RFM features,
export the candidate-count evaluation, fit the requested cluster count, and
save customer-level and profile-level CSVs in `reports/segmentation`.

`tests/test_segmentation.py` will check RFM calculations, feasible cluster
evaluation, and that clustering attaches exactly one meaningful segment label
to every customer.

## Choosing the number of clusters

Use silhouette score as the first quantitative signal: higher is generally
better. Then check whether every segment has a practical size and distinct
business behavior. The default is four clusters, which gives a useful starting
set: Champions, Former High-Value, Potential Customers, and At Risk.

## Run instructions

```powershell
python scripts/run_segmentation.py --input data/raw/online_retail.csv --output reports/segmentation --clusters 4
```

## Interview questions

1. Why is frequency based on distinct invoices rather than transaction rows?
2. Why should RFM features be scaled before K-Means?
3. What does a silhouette score measure, and why should it not be the only cluster-selection criterion?
4. Why is an explicit reference date important when validating a segmentation strategy over time?
5. How would you monitor segment drift after deployment?
