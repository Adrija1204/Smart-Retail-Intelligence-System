FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY app ./app
COPY reports ./reports

EXPOSE 8501

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]

