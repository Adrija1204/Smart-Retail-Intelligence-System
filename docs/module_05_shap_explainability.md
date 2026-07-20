# Module 5: SHAP explainability

## Outcome

Explain the selected purchase model with SHAP values. The global importance
table shows which features influence predictions overall; the local table shows
why a sampled prediction moved up or down.

## Code guide (read before implementation)

1. `TreeExplainer` receives a fitted random forest and score-time feature matrix.
2. `LinearExplainer` receives the fitted logistic-regression pipeline after scaling its features.
3. A helper normalizes SHAP's binary-class output across supported SHAP versions.
4. It creates a feature-importance table from mean absolute SHAP values.
5. It exports one row per customer-feature contribution for a bounded score sample.
6. Training chooses the appropriate explainer for the selected model.

## Interview questions

1. What is the difference between global SHAP importance and a local explanation?
2. Why is SHAP explanation not a causal claim?
3. Which data and model changes would require you to regenerate SHAP outputs?
