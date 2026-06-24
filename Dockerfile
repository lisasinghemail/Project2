FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY isla_coralina_relief_operations.csv .
# Place isla_coralina_infrastructure.csv in this folder before building,
# or upload it in the Streamlit sidebar after the app starts.
# COPY isla_coralina_infrastructure.csv .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
