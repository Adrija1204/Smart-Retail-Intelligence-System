# Module 6: Dashboard and deployment

## Outcome

The Streamlit app reads generated report files and displays business KPIs,
revenue trends, customer segments, model evaluation, top predicted buyers, and
SHAP feature importance. It does not train models interactively, keeping
startup predictable and separating training from serving.

## Code guide (read before implementation)

1. The dashboard resolves report paths relative to the repository root.
2. A safe CSV reader displays an actionable warning if a pipeline output is absent.
3. KPI cards and Plotly charts render EDA and segment reports.
4. Prediction and SHAP sections appear only after training has generated reports.
5. `Dockerfile` installs the package and serves Streamlit on port 8501.
6. `.dockerignore` excludes raw data, artifacts, caches, and secrets from images.
7. The deployment guide supplies local Docker and Streamlit commands plus safeguards.

## Interview questions

1. Why should model training and dashboard serving be separate workflows?
2. What secrets should never be baked into a dashboard image?
3. Which monitoring signals would you add after deployment?
