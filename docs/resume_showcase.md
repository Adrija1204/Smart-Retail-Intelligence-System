# Resume and portfolio showcase

## Project title

**Smart Retail Intelligence System — Customer Analytics and Segmentation**

## Resume bullets

Use one or two of these, depending on available space:

- Built a Python retail-intelligence pipeline that validated and analyzed **1.07M+** transactions, generating revenue, customer, country, product, and monthly-trend insights.
- Engineered RFM features for **5,878** customers and used scaled K-Means clustering with silhouette diagnostics to define four campaign-ready customer segments.
- Identified a Champion segment representing **19% of customers and 72.2% of identified-customer revenue**, enabling retention-focused campaign prioritization.
- Developed tested, reusable Pandas and scikit-learn modules with CSV/Excel schema compatibility and reproducible CLI exports.
- Built a time-aware 30-day purchase-propensity workflow with chronological evaluation, SHAP explainability exports, and a Streamlit dashboard deployment path.

## 30-second interview pitch

“I built a Smart Retail Intelligence System to turn raw transaction records into
campaign decisions. It validates both CSV and Excel retail data, produces EDA
metrics, then engineers Recency, Frequency, and Monetary features for customer
segmentation. I evaluated K-Means cluster counts with silhouette score and used
four actionable segments. The main insight was that Champions made up only 19%
of customers but contributed about 72% of identified-customer revenue, so the
business should prioritize retention and VIP treatment for that group.”

## Two-minute demo flow

1. Open the README and state the business problem: prioritize customer marketing from transaction data.
2. Run the EDA script and show `reports/eda/kpis.csv`.
3. Show `monthly_revenue.csv` and explain seasonal growth into September–November.
4. Show `cluster_evaluation.csv`: two clusters have the best raw silhouette score, while four are selected for richer business actions.
5. Show `segment_profiles.csv` and connect each group to a campaign action.
6. Start the Streamlit dashboard and demonstrate EDA, segments, model metrics, and SHAP importance.

## Strong interview answers

**Why RFM?** It is interpretable, computationally light, and maps directly to
customer lifecycle and marketing actions.

**Why scale before K-Means?** K-Means uses distance. Without log transformation
and standardization, large monetary values would dominate recency and frequency.

**Why choose four clusters when two had a higher silhouette score?** Two is the
best purely geometric split, but four provides a more useful campaign design:
retain Champions, reactivate former high-value customers, grow Potential
Customers, and limit spend on At Risk customers.

**What would you improve for production?** Add scheduled retraining, managed
model artifact storage with versioning, access control, CI/CD, and monitoring
for data, segment, and model-ranking drift.

## Portfolio repository description

“Retail analytics and RFM customer segmentation pipeline built with Python,
Pandas, and scikit-learn. Validates 1M+ transactions, exports business EDA,
and identifies campaign-ready customer groups with K-Means and silhouette
diagnostics.”
