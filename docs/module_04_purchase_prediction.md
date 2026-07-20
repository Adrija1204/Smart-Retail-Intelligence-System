# Module 4: Time-aware purchase prediction

## Outcome

Predict whether an already-known customer will place a completed order in the
next 30 days. Every feature uses transactions strictly before a snapshot date;
the target uses only the following prediction window. This prevents leakage.

## Code guide (read before implementation)

`src/smart_retail/modeling.py` follows these steps:

1. Imports declare the numerical, tabular, preprocessing, model, and metric dependencies.
2. `FEATURE_COLUMNS` keeps training and scoring features aligned.
3. `make_snapshot_dates` creates monthly observation dates after a history period and before the final prediction horizon.
4. `build_purchase_dataset` filters to valid known-customer sales, then loops over snapshot dates.
5. Each historical view is limited to the lookback window ending before its snapshot.
6. It computes recency, distinct-order frequency, spend, and average order value per customer.
7. It searches only the following horizon for the binary future-purchase target.
8. It concatenates snapshot tables into a supervised, time-stamped dataset.
9. `temporal_train_test_split` reserves the newest snapshot dates for evaluation.
10. `build_models` returns a scaled logistic-regression baseline and class-weighted random forest.
11. `evaluate_classifier` computes ROC-AUC, average precision, precision, recall, and F1 from held-out probabilities.
12. `train_and_select_model` fits candidates, evaluates the future holdout, and selects the highest average-precision model.
13. `score_customers` attaches purchase probabilities to an as-of-date feature table.

`scripts/train_purchase_model.py` loads data, creates snapshots, builds the
supervised dataset, trains candidates, exports evaluation and scored customers,
and saves the selected model bundle with Joblib.

## Why average precision is primary

Only a minority of eligible customers purchase in a 30-day window. Accuracy can
therefore look high even when a model finds almost no buyers. Average precision
focuses on the quality of positive-customer ranking.

## Interview questions

1. What is temporal leakage, and how does a snapshot dataset prevent it?
2. Why is a random train/test split inappropriate for future-purchase prediction?
3. Why is average precision often more useful than accuracy for campaign targeting?
4. How would you choose a probability threshold when contacting a customer has a cost?
