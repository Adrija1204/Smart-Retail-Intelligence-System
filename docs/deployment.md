# Deployment guide

## Local run

Generate reports, then start the dashboard:

```powershell
python scripts/run_eda.py --input data/raw/online_retail.csv --output reports/eda
python scripts/run_segmentation.py --input data/raw/online_retail.csv --output reports/segmentation --clusters 4
python scripts/train_purchase_model.py --input data/raw/online_retail.csv --output reports/modeling --model-path models/purchase_model.joblib
streamlit run app/dashboard.py
```

## Docker run

```powershell
docker build -t smart-retail-intelligence .
docker run --rm -p 8501:8501 smart-retail-intelligence
```

Open `http://localhost:8501`.

## Production safeguards

- Store raw data and model artifacts in managed, access-controlled storage.
- Provide secrets through the deployment platform; never commit them.
- Retrain on a schedule and monitor input schema, purchase rate, feature drift, and model ranking quality.
- Restrict dashboard access because customer-level outputs may be sensitive.
