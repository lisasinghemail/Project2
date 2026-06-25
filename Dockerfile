FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY isla_coralina_relief_operations.csv .
COPY isla_coralina_infrastructure.csv .

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]
