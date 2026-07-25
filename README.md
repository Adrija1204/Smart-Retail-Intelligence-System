
A production-oriented retail analytics pipeline that turns 1M+ raw transaction records into customer intelligence, revenue insights, and campaign-ready outputs.

Live demo: https://smart-retail-intelligence-system-8rpxfsxpfg8yhkv2py95tx.streamlit.app/

Overview

This project ingests 1,067,371 retail transactions, validates and cleans them, and runs them through a full analytics pipeline: exploratory analysis, RFM feature engineering, K-Means customer segmentation, a time-aware 30-day purchase propensity model, and SHAP-based explainability. Results are served through an interactive Streamlit dashboard.

Business results
Metric	Result
Valid sales revenue	20.97M
Orders analyzed	40,077
Identified customers	5,878
Average order value	523.31
Champion customers	1,117
Champion revenue share*	72.2%

*Share of revenue attributable to identified customers in the segmentation data.

Pipeline
Raw retail CSV or Excel
Schema validation
Clean valid sales
EDA and KPI exports
RFM features
K-Means segmentation
Time-aware propensitymodel
SHAP explanations
Campaign-ready customergroups
Streamlit dashboard
Features
Ingestion & validation — supports both modern Excel and legacy CSV schemas, with schema checks before any downstream processing.
Revenue cleaning — flags cancellations and computes transaction-level revenue.
EDA & KPIs — exports revenue, order, country, product, and monthly trend analyses.
RFM feature engineering — builds Recency, Frequency, and Monetary customer features.
Segmentation — compares cluster counts using inertia and silhouette score, then produces business-facing segments: Champions, Former High-Value, Potential Customers, and At Risk.
Purchase prediction — predicts 30-day purchase propensity with temporal train/test evaluation.
Explainability — interprets tree-model predictions with SHAP global and local feature contributions.
Delivery — serves all outputs in a Streamlit dashboard, with Docker deployment assets included.
Tech stack

Python · Pandas · NumPy · scikit-learn · pytest · Plotly · SHAP · Streamlit

Dataset

Built on the UCI Online Retail dataset. This implementation targets the dataset's legacy CSV schema.

Place the dataset at:

text
data/raw/online_retail.csv
Quick start
powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_eda.py --input data/raw/online_retail.csv --output reports/eda
python scripts/run_segmentation.py --input data/raw/online_retail.csv --output reports/segmentation --clusters 4
python scripts/train_purchase_model.py --input data/raw/online_retail.csv --output reports/modeling --model-path models/purchase_model.joblib
streamlit run app/dashboard.py
Key insights
The United Kingdom contributes roughly 85% of sales revenue.
Revenue rises sharply from September through November, reflecting seasonal demand.
Champions make up only 19% of identified customers but generate about 72% of identified-customer revenue.
At Risk is the largest segment — low purchase frequency and an average of ~393 days since last purchase. This group should receive low-cost reactivation campaigns rather than high-value incentives.
Repository layout
text
src/smart_retail/       Reusable ingestion, EDA, and segmentation code
scripts/                Reproducible command-line workflows
tests/                  Automated unit tests
docs/                   Module guides, interview questions, and portfolio notes
data/raw/               Local source data (ignored by Git)
reports/                Generated analysis outputs (ignored by Git)
Current scope & roadmap

Implemented: ingestion, validation, EDA, RFM feature engineering, K-Means segmentation, time-aware 30-day purchase prediction, model evaluation, SHAP explainability, Streamlit dashboard, Docker packaging, and automated tests.

Recommended production enhancements: scheduled retraining, managed artifact storage, access control, drift monitoring, CI/CD, and cloud deployment.

Portfolio & interview material

See resume_showcase.md for resume bullets, a 30-second project pitch, a demo flow, and interview talking points.

Learning modules

Each implemented module includes a plain-language code guide and interview questions:

Data ingestion
EDA
Customer segmentation
Purchase prediction
SHAP explainability
Dashboard and deployment
