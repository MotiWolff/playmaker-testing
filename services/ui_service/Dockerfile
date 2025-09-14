FROM python:3.11-slim

WORKDIR /app

COPY services/ui_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ui_service/ /app/

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
